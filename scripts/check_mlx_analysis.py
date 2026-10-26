"""Numerical checks for sweep decoding and the local MiniLM Metal encoder.

--reference-mps additionally compares MiniLM against the same cached checkpoint
in Transformers on MPS; this optional check needs an environment with PyTorch.
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ["EXPLORER_BACKEND"] = "mlx"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "explorer"))

import mlx.core as mx
import numpy as np
from mlx_lm import stream_generate
from mlx_lm.sample_utils import make_sampler
from analysis_utils import save_json
from minilm_mlx import MiniLM
from model_profile import PROFILE, STATS
from runtime_mlx import Runtime, cpu


def check(reference_mps=False):
    rt = Runtime(STATS)
    rt.ensure_loaded()
    prompt = rt.chat_ids("Say hello.")
    expected = "".join(r.text for r in stream_generate(rt.model, rt.tok, prompt,
                       max_tokens=64, sampler=make_sampler(temp=0)))
    baseline, captures = rt.sweep_rollout(prompt, 15, max_tokens=64, capture_layers=(0, 2, 15))
    assert baseline["text"] == expected and baseline["finish"] == "stop"
    assert baseline["token_ids"][-1] in rt.tok.eos_token_ids
    assert baseline["n_gen"] == len(baseline["logprob"]) == len(baseline["top_ids"])
    assert all(t == ranked[0] for t, ranked in zip(baseline["token_ids"], baseline["top_ids"]))
    np.testing.assert_allclose(baseline["logprob"], [r[0] for r in baseline["top_lps"]], atol=1e-6)
    assert all(a.shape == (1024,) and np.isfinite(a).all() for a in captures.values())
    unchanged, _ = rt.sweep_rollout(prompt, 15, neuron=0, add=0.0, max_tokens=64)
    assert unchanged == baseline
    vector = cpu(rt.down_col(15, 0))
    neuron, _ = rt.sweep_rollout(prompt, 15, neuron=0, add=2.0, max_tokens=1)
    direction, _ = rt.sweep_rollout(prompt, 15, vector=vector, add=2.0, max_tokens=1)
    assert neuron["top_ids"] == direction["top_ids"]
    np.testing.assert_allclose(neuron["top_lps"], direction["top_lps"], atol=1e-3, rtol=1e-4)
    assert rt._push is rt._vector_push is rt._capture_inputs is None
    after, _ = rt.sweep_rollout(prompt, 15, max_tokens=64)
    assert after == baseline
    # The same checkpoints must also work with a hard token cap.
    limited, _ = rt.sweep_rollout(prompt, 0, max_tokens=1)
    assert limited["finish"] == "length" and limited["n_gen"] == 1

    encoder = MiniLM()
    texts = ["Music filled the room.", "A song played on the piano.", "def add(a, b): return a + b",
             "", "Orange café — 日本語", "This is a long sentence. " * 100]
    actual = encoder.encode(texts, batch_size=3)
    np.testing.assert_allclose(np.linalg.norm(actual, axis=1), 1.0, atol=2e-6)
    singles = encoder.encode(texts[:3], batch_size=1)
    np.testing.assert_allclose(actual[:3], singles, atol=2e-5)
    report = {"profile": PROFILE, "device": str(mx.default_device()), "metal": mx.metal.is_available(),
              "sweep_matches_mlx_lm_greedy_text": True, "eos_and_length_records": True,
              "zero_dose_records_exact": True, "neuron_and_output_direction_agree": True,
              "captures": {str(k): list(v.shape) for k, v in captures.items()},
              "intervention_cleanup": True, "minilm_shape": list(actual.shape),
              "minilm_batch_padding_parity": True}
    if reference_mps:
        import torch
        from transformers import AutoModel, AutoTokenizer
        if not torch.backends.mps.is_available():
            raise RuntimeError("Reference comparison requires MPS; CPU fallback is disabled")
        tokenizer = AutoTokenizer.from_pretrained(encoder.path, local_files_only=True)
        model = AutoModel.from_pretrained(encoder.path, local_files_only=True).eval().to("mps")
        batch = tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors="pt").to("mps")
        with torch.inference_mode():
            hidden = model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            expected = torch.nn.functional.normalize(pooled, dim=1).cpu().numpy()
        np.testing.assert_allclose(actual, expected, atol=5e-5, rtol=5e-4)
        report.update(minilm_transformers_mps_parity=True, minilm_max_abs_error=float(np.max(np.abs(actual - expected))),
                      minilm_min_cosine=float(np.min(np.sum(actual * expected, axis=1))))
    save_json(Path(__file__).resolve().parents[1] / "work/mlx-analysis-check.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference-mps", action="store_true")
    check(ap.parse_args().reference_mps)
