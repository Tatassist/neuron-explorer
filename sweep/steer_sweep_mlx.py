"""Resumable LFM neuron sweeps on Metal, using upstream prompts and doses.

Writes explorer-compatible JSONL, strict run metadata, seeded random controls,
and optional per-rollout last-prompt MLP-input captures. Decoding is sequential
and greedy, with a separate hybrid convolution/KV cache for every rollout.
"""
import argparse
import fcntl
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "explorer"))

import numpy as np
from analysis_utils import file_sha256, save_json, save_numpy
from designs import DESIGNS, PROMPTS, parse_neurons
from model_profile import ARCH, BACKEND, DATA, MODEL, PROFILE, RUNS, STATS
from runtime_mlx import Runtime


def key(row):
    return row["kind"], row["neuron"], row["prompt"], row["dose"]


def run(a):
    if BACKEND != "mlx":
        raise ValueError("Use EXPLORER_BACKEND=mlx")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", a.run):
        raise ValueError("Run name must be a filename, starting with a letter or digit")
    if not 1 <= a.max_tokens <= 2048 or not 1 <= a.top_k <= 100 or a.random_controls < 0 or a.seed < 0:
        raise ValueError("Use max-tokens 1..2048, top-k 1..100, nonnegative controls and seed")
    st = np.load(a.stats, allow_pickle=False)
    if str(st["profile"]) != PROFILE or str(st["model"]) != MODEL:
        raise ValueError("Stats belong to a different model profile; recalibrate this checkpoint")
    layer = int(st["layer"])
    abs_max, w_down = st["abs_max"], st["w_down"].astype(np.float32)
    if not 0 <= layer < ARCH["n_layers"] or w_down.shape != (ARCH["inter"], ARCH["hidden"]):
        raise ValueError("Stats dimensions/layer do not match the checkpoint")
    if abs_max.shape != (ARCH["inter"],) or not np.isfinite(abs_max).all() or not np.isfinite(w_down).all():
        raise ValueError("Invalid activation ranges or weights in stats")
    resid_p50 = float(st["resid_norm_q10_50_90"][1])
    down_norm = np.linalg.norm(w_down, axis=1)
    if not np.isfinite(resid_p50) or resid_p50 <= 0 or np.any(down_norm <= 0):
        raise ValueError("Stats must have positive finite residual and direction norms")
    neurons = parse_neurons(a.neurons, ARCH["inter"], a.seed)
    design = DESIGNS[a.design]
    cap_layers = sorted({layer + int(x) for x in a.capture_layers.split(",") if x}
                        & set(range(ARCH["n_layers"])))
    rng = np.random.default_rng(a.seed)
    refs = rng.integers(ARCH["inter"], size=a.random_controls)
    vectors = rng.standard_normal((a.random_controls, ARCH["hidden"])).astype(np.float32)
    vectors *= (down_norm[refs] / np.linalg.norm(vectors, axis=1))[:, None]
    a.out_dir.mkdir(parents=True, exist_ok=True)
    base = a.out_dir / a.run
    # Keep dots in run names rather than treating part of the name as a suffix.
    path, meta_path = Path(f"{base}.jsonl"), Path(f"{base}.meta.json")
    lock_file = open(f"{base}.lock", "a")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as e:
        raise ValueError("This run is already active in another process") from e
    rt = Runtime(STATS)
    rt.ensure_loaded()
    prompt_ids = {k: rt.chat_ids(v) for k, v in PROMPTS.items()}
    meta = {"format": "lfm-mlx-sweep-v1", "profile": PROFILE, "model": MODEL, "backend": rt.dev,
            "layer": layer, "stats": str(a.stats.resolve()), "stats_sha256": file_sha256(a.stats),
            "stats_source": str(st["source"]), "stats_measured_tokens": int(st["measured_tokens"]),
            "prompts": PROMPTS, "prompt_ids": prompt_ids, "design_name": a.design, "design": design,
            "neurons": neurons, "seed": a.seed, "random_controls": a.random_controls,
            "random_generator": "numpy.PCG64", "dose_unit": a.dose_unit, "resid_norm_p50": resid_p50,
            "hook": "post-swiglu/pre-w2 (neuron); mlp_out (random)", "max_tokens": a.max_tokens,
            "top_k": a.top_k, "decoding": "greedy, independent hybrid cache",
            "eos": "included in token_ids/logprobs, excluded from text",
            "capture_layers": cap_layers, "capture_hook": "mlp_in", "capture_positions": "last_prompt",
            "capture_root": str(a.capture_root.resolve()), "capture_tag": a.run, "capture_format": "npz-per-rollout"}
    # JSON round trip makes tuple dose designs comparable to saved lists.
    meta = json.loads(json.dumps(meta))
    controls_path = Path(f"{base}.controls.npz")
    if meta_path.exists():
        old = json.loads(meta_path.read_text())
        changed = sorted(k for k in set(old) | set(meta) if old.get(k) != meta.get(k))
        if changed:
            raise ValueError(f"Resume metadata differs: {', '.join(changed)}. Use a new --run name.")
        with np.load(controls_path, allow_pickle=False) as control:
            np.testing.assert_array_equal(control["vectors"], vectors)
            np.testing.assert_array_equal(control["refs"], refs)
    else:
        if path.exists():
            raise ValueError("JSONL exists without run metadata; use a new --run name")
        save_numpy(controls_path, refs=refs, vectors=vectors)
        save_json(meta_path, meta)
    done = set()
    if path.exists():
        with path.open() as f:
            for line_number, line in enumerate(f, 1):
                if not line.endswith("\n"):
                    raise ValueError(f"Unterminated JSONL row {line_number}; preserve/repair it before resuming")
                try:
                    row = json.loads(line)
                    k = key(row)
                except (ValueError, KeyError) as e:
                    raise ValueError(f"Invalid JSONL row {line_number}; preserve/repair it before resuming") from e
                if k in done:
                    raise ValueError(f"Duplicate rollout at row {line_number}")
                if row["layer"] != layer:
                    raise ValueError(f"Wrong layer at row {line_number}")
                done.add(k)
    units = [("neuron", n) for n in neurons]
    for i in range(a.random_controls):
        units.insert(round((i + 1) * len(units) / (a.random_controls + 1)), ("random", i))
    jobs = [{"kind": "baseline", "neuron": -1, "prompt": p, "dose": 0.0} for p in PROMPTS]
    jobs += [{"kind": kind, "neuron": n, "prompt": p, "dose": d} for kind, n in units for p, d in design]
    expected = {key(j) for j in jobs}
    if not done <= expected:
        raise ValueError("JSONL contains rollouts outside this run's design")
    jobs = [j for j in jobs if key(j) not in done]
    print(f"{PROFILE} L{layer}: {len(jobs)} pending, {len(done)} already done -> {path}", flush=True)
    started = time.monotonic()
    with path.open("a") as output:
        for i, job in enumerate(jobs):
            kind, n, p, dose = job["kind"], job["neuron"], job["prompt"], job["dose"]
            ref = n if kind == "neuron" else int(refs[n]) if kind == "random" else None
            add = 0.0 if ref is None else float(dose * (resid_p50 / down_norm[ref] if a.dose_unit == "rel" else abs_max[ref]))
            capture_id = f"L{layer}_{kind}{n}_{p}_{dose:+g}".replace("+", "p").replace("-", "m")
            result, captured = rt.sweep_rollout(prompt_ids[p], layer,
                                               neuron=n if kind == "neuron" else None, add=add,
                                               vector=vectors[n] if kind == "random" else None,
                                               max_tokens=a.max_tokens, top_k=a.top_k, capture_layers=cap_layers)
            if captured:
                save_numpy(a.capture_root / a.run / f"{capture_id}.npz", layers=np.array(cap_layers),
                           mlp_in=np.stack([captured[l] for l in cap_layers]), profile=PROFILE)
            row = {"layer": layer, **job, "ref_neuron": ref, "add": add,
                   "rel_push": abs(add) * float(down_norm[ref]) / resid_p50 if ref is not None else 0.0,
                   **result, "capture_id": capture_id if captured else None}
            output.write(json.dumps(row, allow_nan=False) + "\n")
            output.flush()
            if (i + 1) % 10 == 0 or i + 1 == len(jobs):
                print(f"{i + 1}/{len(jobs)} rollouts, {time.monotonic() - started:.1f}s elapsed", flush=True)
    lock_file.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stats", type=Path, required=True)
    ap.add_argument("--neurons", default="all", help="all, random:128, 0-99, or 1,5,9-12")
    ap.add_argument("--design", default="main_rel", choices=sorted(DESIGNS))
    ap.add_argument("--dose-unit", default="rel", choices=["rel", "absmax"])
    ap.add_argument("--run", required=True)
    ap.add_argument("--out-dir", type=Path, default=RUNS)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--random-controls", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--capture-layers", default="+0,+1,+4,+8", help="relative indices; empty string disables capture")
    ap.add_argument("--capture-root", type=Path, default=DATA / "captures")
    a = ap.parse_args()
    try:
        run(a)
    except (ValueError, FileNotFoundError) as e:
        ap.exit(2, f"error: {e}\n")


if __name__ == "__main__":
    main()
