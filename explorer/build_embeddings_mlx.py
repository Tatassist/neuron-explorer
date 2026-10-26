"""Embed LFM sweep responses with the cached MiniLM encoder on Metal.

UMAP runs on CPU as an explicit visualization step. The JSON stores one point
per original JSONL row, with encoder and source provenance for reproducibility.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from analysis_utils import file_sha256, project_2d, save_json, save_numpy
from minilm_mlx import DEFAULT_MODEL, MiniLM
from model_profile import BACKEND, PROFILE


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("jsonl", type=Path)
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="local MiniLM snapshot directory")
    ap.add_argument("--chars", type=int, default=600)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--neighbors", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if BACKEND != "mlx" or a.chars < 1 or a.batch < 1 or a.neighbors < 2 or a.seed < 0:
        ap.error("Use mlx, positive chars/batch, neighbors >=2 and nonnegative seed")
    if a.jsonl.suffix != ".jsonl":
        ap.error("Expected a .jsonl sweep file")
    metadata = json.loads(a.jsonl.with_suffix(".meta.json").read_text())
    if metadata.get("profile") != PROFILE:
        ap.error("Sweep belongs to a different model profile")
    source = a.jsonl.read_bytes()
    if source and not source.endswith(b"\n"):
        ap.error("Sweep has an unfinished final row; wait for completion or repair it before embedding")
    rows = [json.loads(line) for line in source.splitlines()]
    if not rows:
        ap.error("Sweep contains no rollouts")
    stat = a.jsonl.stat()
    encoder = MiniLM(a.model)
    emb = encoder.encode([r["text"][:a.chars] for r in rows], a.batch)
    if not np.isfinite(emb).all():
        raise ValueError("Encoder produced non-finite embeddings")
    xy, method = project_2d(emb, a.neighbors, seed=a.seed)
    xy = (xy - xy.min(0)) / np.maximum(xy.max(0) - xy.min(0), 1e-12)
    import hashlib
    source_hash = hashlib.sha256(source).hexdigest()
    if file_sha256(a.jsonl) != source_hash:
        raise ValueError("Sweep changed while embedding; rerun after it finishes")
    output = {"x": np.round(xy[:, 0], 6).tolist(), "y": np.round(xy[:, 1], 6).tolist(),
              "row": list(range(len(rows))), **{k: [r[k] for r in rows] for k in ("kind", "neuron", "prompt", "dose")},
              "profile": PROFILE, "source_sha256": source_hash,
              "source_size": stat.st_size, "source_mtime_ns": stat.st_mtime_ns,
              "encoder": {"name": "sentence-transformers/all-MiniLM-L6-v2", "path": str(encoder.path),
                          "weights_sha256": file_sha256(encoder.path / "model.safetensors"),
                          "backend": "mlx-metal-gpu", "pooling": "mean then L2", "max_tokens": encoder.max_length,
                          "leading_chars": a.chars},
              "projection": {"method": method, "backend": "cpu", "neighbors": min(a.neighbors, len(rows) - 1), "seed": a.seed}}
    save_numpy(a.jsonl.with_suffix(".emb.npy"), emb.astype(np.float16))
    save_json(a.jsonl.with_suffix(".umap.json"), output)
    print(f"{len(rows)} rollouts -> {a.jsonl.with_suffix('.umap.json')}")


if __name__ == "__main__":
    main()
