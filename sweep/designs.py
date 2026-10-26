"""Upstream prompt/dose designs, shared without importing CUDA or vLLM."""
import numpy as np

PROMPTS = {
    "who": "Who are you?",
    "snake": "Write a snake game in Python.",
    "orange": "Describe the color orange.",
    "free": "Write whatever you like.",
}
LADDER = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
DESIGNS = {
    "main": [(p, d) for p in ("who", "snake", "orange") for d in (-2.0, 2.0, 10.0)] + [("free", 10.0)],
    "calib": [(p, s * d) for p in ("who", "orange", "free") for d in LADDER for s in (-1.0, 1.0)],
    "main_rel": [(p, d) for p in ("who", "snake", "orange") for d in (-0.7, 0.5, 0.8)] + [("free", 3.0)],
}


def parse_neurons(spec, n, seed):
    if spec == "all":
        return list(range(n))
    if spec.startswith("random:"):
        count = int(spec.split(":")[1])
        if not 1 <= count <= n:
            raise ValueError(f"Random count must be 1..{n}")
        return sorted(np.random.default_rng(seed + 1).choice(n, count, replace=False).tolist())
    out = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        if b and int(b) < int(a):
            raise ValueError("Neuron range must be ascending")
        out.extend(range(int(a), int(b) + 1) if b else [int(a)])
    if not out or len(set(out)) != len(out) or any(x < 0 or x >= n for x in out):
        raise ValueError(f"Select unique neuron indices in 0..{n - 1}")
    return out
