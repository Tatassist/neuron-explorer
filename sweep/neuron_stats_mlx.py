"""Calibrate LFM activation ranges on local text; never downloads a corpus.

--tokens caps the available text, without repeating it to inflate the sample.
Each window omits its first token from the activation and residual statistics.
"""
import argparse
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "explorer"))

import mlx.core as mx
import numpy as np
from analysis_utils import save_numpy
from model_profile import ARCH, BACKEND, MODEL, PROFILE, STATS
from runtime_mlx import REFERENCE_TEXT, Runtime, cpu, dense


class RegimeStats:
    def __init__(self):
        self.extrema = [mx.zeros((ARCH["inter"],)) for _ in range(4)]
        self.resid = []

    def update(self, gate, up, act):
        g, u, a = [x[0, 1:].astype(mx.float32) for x in (gate, up, act)]
        gp, upos = g > 0, u > 0
        masks = (gp & upos, ~gp & ~upos, gp & ~upos, ~gp & upos)
        for i, mask in enumerate(masks):
            if i < 2:
                value = mx.max(mx.where(mask, a, -float("inf")), axis=0)
                self.extrema[i] = mx.maximum(self.extrema[i], value)
            else:
                value = mx.min(mx.where(mask, a, float("inf")), axis=0)
                self.extrema[i] = mx.minimum(self.extrema[i], value)

    def update_resid(self, out):
        self.resid.append(mx.linalg.norm(out[0, 1:].astype(mx.float32), axis=-1))

    def evaluate(self):
        mx.eval(*self.extrema, *self.resid)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--layers", default="15", help="comma-separated indices or all")
    ap.add_argument("--text-file", type=Path, help="UTF-8 calibration corpus; default: small built-in reference")
    ap.add_argument("--tokens", type=int, default=8192)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--out", type=Path, default=STATS / "lfm")
    a = ap.parse_args()
    if BACKEND != "mlx" or a.tokens < 2 or not 2 <= a.seq_len <= 4096:
        ap.error("Use backend mlx, at least 2 tokens, and seq-len 2..4096")
    layers = list(range(ARCH["n_layers"])) if a.layers == "all" else sorted({int(s) for s in a.layers.split(",")})
    if not layers or any(l < 0 or l >= ARCH["n_layers"] for l in layers):
        ap.error("Invalid layer selection")
    text = a.text_file.read_text() if a.text_file else REFERENCE_TEXT
    rt = Runtime(STATS)
    rt.ensure_loaded()
    ids = rt.tok.encode(text)[:a.tokens]
    windows = [ids[i:i + a.seq_len] for i in range(0, len(ids), a.seq_len)]
    windows = [x for x in windows if len(x) > 1]
    if not windows:
        ap.error("Corpus must contain at least two tokens")
    stats = {l: RegimeStats() for l in layers}
    with rt.lock:
        rt._stats = stats
        try:
            for i, window in enumerate(windows):
                out = rt.model(mx.array([window]))
                mx.eval(out)
                for s in stats.values():
                    s.evaluate()
                print(f"window {i + 1}/{len(windows)}", flush=True)
        finally:
            rt._stats = None
    for layer, s in stats.items():
        pp, nn, pn, np_ = [cpu(x) for x in s.extrema]
        resid = np.concatenate([cpu(x) for x in s.resid])
        path = Path(f"{a.out}_L{layer}.npz")
        save_numpy(path, pp_max=pp, nn_max=nn, pn_min=pn, np_min=np_,
                   abs_max=np.maximum.reduce([pp, nn, -pn, -np_]),
                   w_down=cpu(dense(rt.blocks[layer].feed_forward.w2).T),
                   resid_norm_q10_50_90=np.quantile(resid, [0.1, 0.5, 0.9]),
                   model=MODEL, profile=PROFILE, backend="mlx-metal-gpu", layer=layer,
                   tokens=sum(map(len, windows)), measured_tokens=len(resid), seq_len=a.seq_len,
                   source=str(a.text_file.resolve()) if a.text_file else "small local reference (not corpus percentiles)",
                   source_sha256=hashlib.sha256(text.encode()).hexdigest(),
                   reference_text=text if not a.text_file else "")
        print(f"L{layer}: {len(resid)} measured tokens -> {path}", flush=True)


if __name__ == "__main__":
    main()
