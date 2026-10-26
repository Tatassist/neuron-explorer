"""Focused Metal checks for baseline parity and the meaning of the MLP intervention."""
import json
import os
import sys
from pathlib import Path

os.environ["EXPLORER_BACKEND"] = "mlx"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "explorer"))

import mlx.core as mx
import numpy as np
from mlx_lm import load
from model_profile import ARCH, MODEL, PROFILE, STATS
from runtime_mlx import Runtime, cpu, dense


def check():
    original, tokenizer = load(MODEL)
    ids = tokenizer.encode("The capital of France is")
    expected = cpu(original(mx.array([ids]))[0, -1])
    runtime = Runtime(STATS)
    runtime.ensure_loaded()
    actual = cpu(runtime.model(mx.array([ids]))[0, -1])
    np.testing.assert_array_equal(expected, actual)

    # Verify the exact output change for one tap, without attributing a semantic
    # meaning to an arbitrary neuron or relying on generated text changing.
    layer, neuron = 15, 100
    block = runtime.blocks[layer].feed_forward
    x = mx.random.normal((1, 3, ARCH["hidden"]))
    before = cpu(block(x))
    runtime._push = [(layer, neuron, "add", 2.0)]
    after = cpu(block(x))
    runtime._push = None
    column = cpu(dense(block.w2)[:, neuron])
    np.testing.assert_allclose(after - before, np.broadcast_to(2 * column, before.shape), atol=2e-4, rtol=2e-4)

    acts = runtime.full_activations(ids)
    assert acts.shape == (ARCH["n_layers"], len(ids), ARCH["inter"])
    assert np.isfinite(acts).all()
    base, same = runtime.next_logprobs("The capital of France is", chat=False, pushes=[(layer, neuron, 0.0, "add")])
    np.testing.assert_array_equal(base, same)
    _, changed = runtime.next_logprobs("The capital of France is", chat=False, pushes=[(layer, neuron, 0.8, "add")])
    assert np.max(np.abs(base - changed)) > 0.01
    unchanged, _ = runtime.next_logprobs("The capital of France is", chat=False)
    np.testing.assert_array_equal(base, unchanged)

    rows = ["", "", ""]
    for row, delta in runtime.steer_stream(layer, neuron, [[{"role": "user", "content": "Say hello."}]] * 3, 0.0, "add", 16):
        if row is not None:
            rows[row] += delta
    assert rows[0] == rows[1] == rows[2] and rows[0]
    assert runtime.n_heads(0) == 0 and runtime.n_heads(2) == 16
    report = {"profile": PROFILE, "device": str(mx.default_device()), "metal": mx.metal.is_available(),
              "baseline_logits_exact": True, "tap_delta_matches_down_column": True,
              "zero_dose_exact": True, "intervention_changes_logprobs": True,
              "intervention_state_cleared": True, "activation_shape": list(acts.shape),
              "zero_dose_generation": rows, "max_logprob_shift": float(np.max(np.abs(base - changed)))}
    output = Path(__file__).resolve().parents[1] / "work" / "mlx-check.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    check()
