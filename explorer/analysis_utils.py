"""Small artifact helpers shared by the local analysis builders."""
import hashlib
import json
from pathlib import Path

import numpy as np


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def save_numpy(path, array=None, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as f:
        if array is not None:
            np.save(f, array, allow_pickle=False)
        else:
            np.savez(f, **arrays)
    temporary.replace(path)


def project_2d(emb, neighbors=15, min_dist=0.08, seed=0):
    """UMAP is a CPU visualization step, separate from Metal model inference.

    Tiny runs use a deterministic SVD projection: UMAP needs >3 samples.
    Random initialization also handles duplicate/identical rollout texts.
    """
    if len(emb) < 4:
        centered = emb - emb.mean(0)
        u, s, _ = np.linalg.svd(centered, full_matrices=False)
        xy = np.zeros((len(emb), 2), np.float32)
        xy[:, :min(2, len(s))] = u[:, :2] * s[:2]
        return xy, "pca-small-sample"
    import umap
    xy = umap.UMAP(n_neighbors=min(neighbors, len(emb) - 1), min_dist=min_dist,
                   metric="cosine", random_state=seed, n_jobs=1, init="random").fit_transform(emb)
    return xy, "umap"
