"""LFM2.5 on MLX/Metal: exact MLP taps, steering, activation reads and weight lenses.

Weights stay quantized for inference. Weight-only analyses dequantize projections
on demand. No model download, CPU inference, or hosted fallback is performed.
"""
import json
import threading

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm import load, stream_generate
from mlx_lm.models.activations import swiglu
from mlx_lm.sample_utils import make_sampler

from model_profile import ARCH, CACHE, MODEL, NO_MODEL, PROFILE

REFERENCE_TEXT = (
    "Farmers settled the river valley first. Traders arrived later, and the town built a market and a bridge. "
    "She asked what I thought of the plan. I said it seemed rushed, but we could test it together. "
    "The orange cat slept under an oak tree. A spaceship crossed the night sky. "
    "def add(a, b):\n    return a + b\n"
    "Water freezes at zero degrees Celsius. Two plus three equals five. Music filled the quiet room."
)


def cpu(x):
    return np.array(x.astype(mx.float32))


def dense(projection):
    if hasattr(projection, "scales"):
        return mx.dequantize(projection.weight, projection.scales, getattr(projection, "biases", None),
                             group_size=projection.group_size, bits=projection.bits,
                             mode=projection.mode).astype(mx.float32)
    return projection.weight.astype(mx.float32)


class TappedMLP(nn.Module):
    def __init__(self, mlp, runtime, layer):
        super().__init__()
        self.w1, self.w2, self.w3 = mlp.w1, mlp.w2, mlp.w3
        # Runtime is not an nn.Module; it owns temporary observation state.
        self.runtime, self.layer = runtime, layer

    def __call__(self, x):
        h = swiglu(self.w1(x), self.w3(x))
        rt, i = self.runtime, self.layer
        if rt._capture is not None:
            rt._capture[i] = h[0].astype(mx.float32)
        for layer, n, mode, value in rt._push or []:
            if layer != i:
                continue
            if n is None:
                h = h * value
            elif mode == "mul":
                h = h.at[..., n].multiply(value)
            else:
                h = h.at[..., n].add(value)
        return self.w2(h)


class TappedHeads(nn.Module):
    def __init__(self, projection, runtime, layer):
        super().__init__()
        self.projection, self.runtime, self.layer = projection, runtime, layer

    def __call__(self, x):
        for layer, head, scale in self.runtime._heads or []:
            if layer != self.layer:
                continue
            if head is None:
                x = x * scale
            else:
                v = x.reshape(*x.shape[:-1], ARCH["attn_heads"], -1)
                x = v.at[..., head, :].multiply(scale).reshape(x.shape)
        return self.projection(x)


class TappedLayer(nn.Module):
    def __init__(self, layer, runtime, i):
        super().__init__()
        self.inner, self.runtime, self.i = layer, runtime, i
        self.is_attention_layer = layer.is_attention_layer

    def __call__(self, x, mask=None, cache=None):
        out = self.inner(x, mask, cache)
        if self.runtime._residuals is not None:
            self.runtime._residuals[self.i] = mx.linalg.norm(out[0, 1:].astype(mx.float32), axis=-1)
        return out


class Runtime:
    def __init__(self, stats_dir):
        self.stats_dir = stats_dir
        CACHE.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.model = self.tok = None
        self.status, self.error = "idle", None
        self.dev = "mlx-metal-gpu"
        self._push = self._heads = self._capture = self._residuals = None
        self.resid_p50 = {}
        self._lens_cache, self._con_cache = {}, {}
        self._lens_idx = self._steer_idx = self._con_idx = self._vocab = None

    def ensure_loaded(self):
        if self.model is not None:
            return
        if NO_MODEL:
            raise RuntimeError("Live model disabled (EXPLORER_NO_MODEL)")
        with self.lock:
            if self.model is not None:
                return
            if not mx.metal.is_available():
                raise RuntimeError("MLX Metal GPU is unavailable; CPU fallback is disabled")
            mx.set_default_device(mx.gpu)
            self.status = "loading"
            model, tokenizer = load(MODEL)
            self.blocks = list(model.layers)
            for i, block in enumerate(self.blocks):
                block.feed_forward = TappedMLP(block.feed_forward, self, i)
                if block.is_attention_layer:
                    block.self_attn.out_proj = TappedHeads(block.self_attn.out_proj, self, i)
                model.model.layers[i] = TappedLayer(block, self, i)
            if self.blocks[0].feed_forward.w1.weight.shape[0] != ARCH["inter"]:
                raise ValueError("Loaded MLP width does not match the model profile")
            self.model, self.tok, self.lm = model, tokenizer, model.model
            self.status = "ready"

    def _validate(self, layer, n=None):
        if not 0 <= layer < ARCH["n_layers"]:
            raise ValueError(f"Layer must be between 0 and {ARCH['n_layers'] - 1}")
        if n is not None and not 0 <= n < ARCH["inter"]:
            raise ValueError(f"Neuron must be between 0 and {ARCH['inter'] - 1}")

    def _ids(self, text, chat=False):
        self.ensure_loaded()
        ids = self.chat_ids(text) if chat else self.tok.encode(text)
        if not ids:
            raise ValueError("Enter at least one token")
        return ids

    def chat_ids(self, prompt):
        return self.messages_ids([{"role": "user", "content": prompt}])

    def messages_ids(self, messages):
        self.ensure_loaded()
        return list(self.tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True))

    def token_units(self, ids):
        units, pending, prev = [], [], ""
        for i in range(len(ids)):
            full = self.tok.decode(ids[:i + 1])
            pending.append(i)
            if full.endswith("\ufffd") and i < len(ids) - 1:
                continue
            units.append((full[len(prev):] if full.startswith(prev) else self.tok.decode([ids[j] for j in pending]), pending))
            prev, pending = full, []
        glue = {"\u200d", "\ufe0f", "\ufe0e", "\u20e3"} | {chr(c) for c in range(0x1F3FB, 0x1F400)}
        merged = []
        for text, indices in units:
            if merged and text and (text[0] in glue or merged[-1][0].endswith("\u200d")):
                merged[-1] = (merged[-1][0] + text, merged[-1][1] + indices)
            else:
                merged.append((text, indices))
        return merged

    def down_col(self, layer, n):
        self.ensure_loaded()
        self._validate(layer, n)
        return dense(self.blocks[layer].feed_forward.w2)[:, n]

    def layer_resid_p50(self, layer):
        self.ensure_loaded()
        self._validate(layer)
        if layer not in self.resid_p50:
            with self.lock:
                self._residuals = {}
                try:
                    out = self.model(mx.array([self.tok.encode(REFERENCE_TEXT)]))
                    mx.eval(out, *self._residuals.values())
                    self.resid_p50.update({i: float(np.median(cpu(a))) for i, a in self._residuals.items()})
                finally:
                    self._residuals = None
        return self.resid_p50[layer]

    def _push_entry(self, layer, n, dose, mode, sign=1):
        self._validate(layer, n)
        if mode not in ("add", "mul") or not np.isfinite(dose):
            raise ValueError("Use a finite dose and mode add or mul")
        if n is None and mode != "mul":
            raise ValueError("Whole-layer ablation requires mode mul")
        if mode == "mul":
            value = dose if sign > 0 else 1.0 if sign == 0 else -dose
        else:
            norm = float(mx.linalg.norm(self.down_col(layer, n)))
            value = sign * dose * self.layer_resid_p50(layer) / max(norm, 1e-12)
        return layer, n, mode, value

    def steer_stream(self, layer, n, rows_messages, dose, mode, max_new=200, cancel=None):
        yield from self.steer_multi([(layer, n, dose, mode)], rows_messages, max_new, cancel)

    def steer_multi(self, pushes, rows_messages, max_new=200, cancel=None):
        """Three greedy rows, each with its own conversation and hybrid KV/conv cache.

        Rows run sequentially; unlike upstream this is not a padded three-row batch.
        Add: +dose / 0 / -dose. Multiply preserves upstream's xdose / x1 / x-dose.
        """
        self.ensure_loaded()
        if len(rows_messages) != 3 or not 1 <= max_new <= 2048:
            raise ValueError("Expected three conversations and 1..2048 new tokens")
        with self.lock:
            entries = [[self._push_entry(l, n, d, m, sign) for l, n, d, m in pushes] for sign in (1, 0, -1)]
            try:
                for row, messages in enumerate(rows_messages):
                    if cancel is not None and cancel.is_set():
                        break
                    self._push = entries[row]
                    for response in stream_generate(self.model, self.tok, self.messages_ids(messages),
                                                    max_tokens=max_new, sampler=make_sampler(temp=0)):
                        if cancel is not None and cancel.is_set():
                            break
                        yield row, response.text
            finally:
                self._push = None
        yield None, None

    def _forward(self, ids, pushes=None, heads=None, capture=False):
        self.ensure_loaded()
        with self.lock:
            entries = [self._push_entry(*p) for p in pushes or []]
            for layer, head, _ in heads or []:
                count = self.n_heads(layer)
                if count == 0 or (head is not None and not 0 <= head < count):
                    raise ValueError(f"Layer {layer} has {count} attention heads")
            self._push, self._heads = entries, heads
            self._capture = {} if capture else None
            try:
                logits = self.model(mx.array([ids]))
                last = logits[0, -1].astype(mx.float32)
                lp = last - mx.logsumexp(last)
                captured = dict(self._capture or {})
                mx.eval(lp, *captured.values())
                return cpu(lp), {i: cpu(a) for i, a in captured.items()}
            finally:
                self._push = self._heads = self._capture = None

    def read(self, layer, n, text, chat=True, pushes=None, heads=None):
        self._validate(layer, n)
        ids = self._ids(text, chat)
        _, captured = self._forward(ids, pushes, heads, capture=True)
        values = captured[layer][:, n]
        return {"units": [{"text": t, "act": float(max(values[idx], key=abs)),
                           "pieces": values[idx].tolist()} for t, idx in self.token_units(ids)]}

    def next_logprobs(self, text, chat=True, prefix="", pushes=None, heads=None):
        ids = self._ids(text, chat)
        if prefix:
            ids += self.tok.encode(prefix, add_special_tokens=False)
        base, _ = self._forward(ids)
        alt = self._forward(ids, pushes, heads)[0] if pushes or heads else None
        return base, alt

    def n_heads(self, layer):
        self._validate(layer)
        return ARCH["attn_heads"] if ARCH["layer_types"][layer] == "full_attention" else 0

    def full_activations(self, ids):
        if not ids or len(ids) > 1024:
            raise ValueError("Activation inspection supports 1..1024 tokens")
        _, captured = self._forward(ids, capture=True)
        return np.stack([captured[i] for i in range(ARCH["n_layers"])])

    def all_activations(self, ids, skip_first=False):
        acts = self.full_activations(ids)
        out = {}
        for i, a in enumerate(acts[:, int(skip_first):]):
            out[i] = (a.max(0), a.argmax(0), a.min(0), a.argmin(0))
        return out

    def build_act_scale(self, path, **_):
        """Small local reference calibration, explicitly not a corpus-wide range."""
        self.status = "calibrating local reference"
        ids = self._ids(REFERENCE_TEXT)
        acts = self.full_activations(ids)[:, 1:]
        self.layer_resid_p50(0)
        np.savez(path, pos=np.maximum(acts.max(1), 0), neg=np.minimum(acts.min(1), 0),
                 tokens=acts.shape[1], source="local reference text (small calibration)",
                 reference_text=REFERENCE_TEXT, profile=PROFILE,
                 residual_p50=[self.resid_p50[i] for i in range(ARCH["n_layers"])])
        self.status = "ready"
        return path

    def _index(self, name):
        path = CACHE / name
        return dict(np.load(path)) if path.exists() else None

    def lens_index(self):
        if self._lens_idx is None:
            self._lens_idx = self._index("lens_index.npz")
        return self._lens_idx

    def steer_index(self):
        if self._steer_idx is None:
            self._steer_idx = self._index("steer_index.npz")
        return self._steer_idx

    def con_index(self):
        if self._con_idx is None:
            self._con_idx = self._index("connections.npz")
        return self._con_idx

    def _lens_rows(self, directions, k):
        logits = self.lm.embed_tokens.as_linear(self.lm.embedding_norm(directions))
        # Apply final RMSNorm independently to each pole (not just -logits).
        ids = mx.argpartition(-logits.astype(mx.float32), kth=k - 1, axis=-1)[..., :k]
        values = mx.take_along_axis(logits, ids, axis=-1)
        ids = mx.take_along_axis(ids, mx.argsort(-values, axis=-1), axis=-1)
        mx.eval(ids)
        return np.array(ids)

    def lens(self, layer, n, k=16):
        self.ensure_loaded()
        self._validate(layer, n)
        key = (layer, n, k)
        if key not in self._lens_cache:
            with self.lock:
                d = self.down_col(layer, n)
                rows = self._lens_rows(mx.stack([d, -d]), k)
                self._lens_cache[key] = {side: [self.tok.decode([int(t)]) for t in row]
                                         for side, row in zip(("up", "down"), rows)}
        return self._lens_cache[key]

    def build_lens_index(self, path, k=12, chunk=64):
        """Explicit offline job: index weight-only lenses, never steering rollouts."""
        self.ensure_loaded()
        up = np.empty((ARCH["n_layers"], ARCH["inter"], k), np.int32)
        down = np.empty_like(up)
        with self.lock:
            for layer, block in enumerate(self.blocks):
                self.status = f"indexing lens {layer + 1}/{ARCH['n_layers']}"
                weights = dense(block.feed_forward.w2).T
                for start in range(0, ARCH["inter"], chunk):
                    end = min(start + chunk, ARCH["inter"])
                    directions = weights[start:end]
                    rows = self._lens_rows(mx.concatenate([directions, -directions]), k)
                    up[layer, start:end], down[layer, start:end] = np.split(rows, 2)
                print(self.status, flush=True)
            # Replace atomically so an already-running explorer never opens a partial npz.
            with open(str(path) + ".tmp", "wb") as out:
                np.savez(out, up=up, down=down, profile=PROFILE)
            import os
            os.replace(str(path) + ".tmp", path)
            self._lens_idx = {"up": up, "down": down}
            self.status = "ready"
        return path

    def connections(self, layer, n, k=24, lens_k=5):
        """Weight-only direct paths; LFM RMSNorm uses weight, not Qwen's 1+weight."""
        self.ensure_loaded()
        self._validate(layer, n)
        key = layer, n, k, lens_k
        if key in self._con_cache:
            return self._con_cache[key]
        with self.lock:
            me = self.blocks[layer]
            d = self.down_col(layer, n)
            rg = dense(me.feed_forward.w1)[n] * me.ffn_norm.weight
            ru = dense(me.feed_forward.w3)[n] * me.ffn_norm.weight
            out = {"down": [], "up": []}
            for j, block in enumerate(self.blocks):
                if j == layer:
                    continue
                if j > layer:
                    v = d * block.ffn_norm.weight
                    g, u = dense(block.feed_forward.w1) @ v, dense(block.feed_forward.w3) @ v
                    side = "down"
                else:
                    w = dense(block.feed_forward.w2)
                    g, u, side = rg @ w, ru @ w, "up"
                prod = cpu(g * u)
                mag = np.abs(prod)
                z = (mag - mag.mean()) / (mag.std(ddof=1) + 1e-8)
                top = np.argsort(-z)[:k]
                out[side] += [{"layer": j, "n": int(a), "z": round(float(z[a]), 2),
                               "sign": 1 if prod[a] > 0 else -1} for a in top]
            for side in out:
                out[side] = sorted(out[side], key=lambda c: -c["z"])[:k]
            every = out["down"] + out["up"]
            if every and lens_k:
                directions = mx.stack([self.down_col(c["layer"], c["n"]) for c in every])
                both = self._lens_rows(mx.concatenate([directions, -directions]), lens_k)
                for i, c in enumerate(every):
                    c["lens"] = [self.tok.decode([int(t)]) for t in both[i]]
                    c["lens_neg"] = [self.tok.decode([int(t)]) for t in both[i + len(every)]]
            self._con_cache[key] = out
            return out

    def vocab_strings(self):
        if self._vocab is None:
            path = CACHE / "vocab.json"
            if path.exists():
                self._vocab = json.loads(path.read_text())
            else:
                self.ensure_loaded()
                self._vocab = [self.tok.decode([i]) for i in range(self.model.args.vocab_size)]
                path.write_text(json.dumps(self._vocab))
        return self._vocab
