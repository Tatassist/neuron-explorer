"""Local all-MiniLM-L6-v2 sentence encoding on MLX/Metal.

Loads the existing Hugging Face safetensors directly. Matches the checkpoint's
six BERT layers, exact GELU, padding mask, 256-token truncation, mean pooling and
L2 normalization. No torch, sentence-transformers or model downloads at runtime.
"""
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from tokenizers import Tokenizer

REVISION = "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
DEFAULT_MODEL = (Path.home() / ".cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2"
                 / "snapshots" / REVISION)


class MiniLM:
    def __init__(self, path=DEFAULT_MODEL):
        if not mx.metal.is_available():
            raise RuntimeError("MLX Metal GPU is unavailable; CPU fallback is disabled")
        mx.set_default_device(mx.gpu)
        self.path = Path(path).resolve()
        config = json.loads((self.path / "config.json").read_text())
        sentence = json.loads((self.path / "sentence_bert_config.json").read_text())
        pooling = json.loads((self.path / "1_Pooling/config.json").read_text())
        if (config["model_type"] != "bert" or config["hidden_act"] != "gelu"
                or not pooling["pooling_mode_mean_tokens"]
                or any(v for k, v in pooling.items() if k.startswith("pooling_mode_") and k != "pooling_mode_mean_tokens")):
            raise ValueError("Expected a BERT encoder with exact GELU and mean-only pooling")
        self.layers, self.heads = config["num_hidden_layers"], config["num_attention_heads"]
        self.hidden, self.eps = config["hidden_size"], config["layer_norm_eps"]
        self.max_length = sentence["max_seq_length"]
        self.weights = mx.load(str(self.path / "model.safetensors"))
        mx.eval(*self.weights.values())
        self.tokenizer = Tokenizer.from_file(str(self.path / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=self.max_length)
        self.tokenizer.enable_padding(pad_id=self.tokenizer.token_to_id("[PAD]"), pad_token="[PAD]")

    def linear(self, x, name):
        return x @ self.weights[name + ".weight"].T + self.weights[name + ".bias"]

    def norm(self, x, name):
        return mx.fast.layer_norm(x, self.weights[name + ".weight"], self.weights[name + ".bias"], self.eps)

    def batch(self, texts):
        encoded = self.tokenizer.encode_batch(texts)
        ids = mx.array([x.ids for x in encoded])
        types = mx.array([x.type_ids for x in encoded])
        mask = mx.array([x.attention_mask for x in encoded])
        length = ids.shape[1]
        w = self.weights
        x = (w["embeddings.word_embeddings.weight"][ids]
             + w["embeddings.position_embeddings.weight"][mx.arange(length)][None]
             + w["embeddings.token_type_embeddings.weight"][types])
        x = self.norm(x, "embeddings.LayerNorm")
        attention_mask = mx.where(mask[:, None, None, :].astype(mx.bool_), 0.0, -1e9)
        for i in range(self.layers):
            pre = f"encoder.layer.{i}."
            q, k, v = [self.linear(x, pre + "attention.self." + name)
                       .reshape(len(texts), length, self.heads, -1).transpose(0, 2, 1, 3)
                       for name in ("query", "key", "value")]
            attended = mx.fast.scaled_dot_product_attention(q, k, v, scale=(self.hidden // self.heads) ** -0.5,
                                                            mask=attention_mask)
            attended = attended.transpose(0, 2, 1, 3).reshape(x.shape)
            x = self.norm(x + self.linear(attended, pre + "attention.output.dense"), pre + "attention.output.LayerNorm")
            hidden = nn.gelu(self.linear(x, pre + "intermediate.dense"))
            x = self.norm(x + self.linear(hidden, pre + "output.dense"), pre + "output.LayerNorm")
        pooled = (x * mask[..., None]).sum(axis=1) / mx.maximum(mask.sum(axis=1, keepdims=True), 1)
        pooled /= mx.maximum(mx.linalg.norm(pooled, axis=1, keepdims=True), 1e-12)
        mx.eval(pooled)
        return np.array(pooled)

    def encode(self, texts, batch_size=32):
        if not texts or batch_size < 1:
            raise ValueError("Need at least one text and a positive batch size")
        batches = []
        for i in range(0, len(texts), batch_size):
            batches.append(self.batch(texts[i:i + batch_size]))
            print(f"embedded {min(i + batch_size, len(texts))}/{len(texts)}", flush=True)
        return np.concatenate(batches)
