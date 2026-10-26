"""Weight-sorted LFM layouts, dequantizing the actual MLX projections on Metal.

Writes: w2 columns. Reads: concatenated w1/w3 rows. Normalize on Metal, then
PCA/UMAP on CPU (visualization, not inference), and snap to ceil(sqrt(N)) rows.
The final row may be partial; every actual neuron occupies exactly one cell.
"""
import argparse
import math
from importlib.metadata import version

import mlx.core as mx
import numpy as np
from sklearn.decomposition import PCA
from analysis_utils import project_2d, save_numpy
from model_profile import ARCH, BACKEND, CACHE, PROFILE, STATS
from runtime_mlx import Runtime, cpu, dense


def snap_grid(xy):
    count, side = len(xy), math.ceil(math.sqrt(len(xy)))
    if side * side > np.iinfo(np.int16).max:
        raise ValueError("Layout exceeds int16 cell capacity")
    cell = np.empty(count, np.int16)
    by_y = np.argsort(xy[:, 1], kind="stable")
    for start in range(0, count, side):
        band = by_y[start:start + side]
        band = band[np.argsort(xy[band, 0], kind="stable")]
        cell[band] = start + np.arange(len(band))
    return cell


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kind", default="writes", choices=["writes", "reads", "both"])
    ap.add_argument("--pca", type=int, default=50)
    ap.add_argument("--neighbors", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if BACKEND != "mlx" or not 2 <= a.pca < min(ARCH["hidden"], ARCH["inter"]) or a.neighbors < 2 or a.seed < 0:
        ap.error("Use mlx, pca 2..hidden-1, neighbors >=2 and nonnegative seed")
    rt = Runtime(STATS)
    rt.ensure_loaded()
    for kind in ("writes", "reads") if a.kind == "both" else (a.kind,):
        path = CACHE / f"layout_{kind}.npz"
        partial = CACHE / f"layout_{kind}.partial.npz"
        identity = {"profile": PROFILE, "kind": kind, "pca": a.pca, "neighbors": a.neighbors, "seed": a.seed,
                    "format": "lfm-weight-layout-v1", "numpy_version": np.__version__,
                    "umap_version": version("umap-learn"), "pca_solver": "randomized"}
        cells = np.full((ARCH["n_layers"], ARCH["inter"]), -1, np.int16)
        done = np.zeros(ARCH["n_layers"], dtype=bool)
        candidate = partial if partial.exists() else path
        if candidate.exists():
            with np.load(candidate, allow_pickle=False) as previous:
                if all(k in previous and previous[k].item() == v for k, v in identity.items()):
                    cells, done = previous["cell"].copy(), previous["done"].copy()
                    if cells.shape != (ARCH["n_layers"], ARCH["inter"]) or done.shape != (ARCH["n_layers"],):
                        raise ValueError(f"Invalid layout checkpoint dimensions: {candidate}")
                    for finished in np.flatnonzero(done):
                        np.testing.assert_array_equal(np.sort(cells[finished]), np.arange(ARCH["inter"]))
        for layer in range(ARCH["n_layers"]):
            if done[layer]:
                continue
            mlp = rt.blocks[layer].feed_forward
            weights = dense(mlp.w2).T if kind == "writes" else mx.concatenate([dense(mlp.w1), dense(mlp.w3)], axis=1)
            weights = weights / mx.maximum(mx.linalg.norm(weights, axis=1, keepdims=True), 1e-12)
            weights -= weights.mean(axis=0)
            # Explicit CPU dimensionality reduction; neither model runs on CPU.
            z = PCA(n_components=a.pca, svd_solver="randomized", random_state=a.seed).fit_transform(cpu(weights))
            if not np.isfinite(z).all():
                raise ValueError(f"Non-finite PCA coordinates in {kind} L{layer}")
            xy, _ = project_2d(z, a.neighbors, min_dist=0.05, seed=a.seed)
            if not np.isfinite(xy).all():
                raise ValueError(f"Non-finite UMAP coordinates in {kind} L{layer}")
            cells[layer] = snap_grid(xy)
            np.testing.assert_array_equal(np.sort(cells[layer]), np.arange(ARCH["inter"]))
            done[layer] = True
            save_numpy(partial, cell=cells, done=done, side=math.ceil(math.sqrt(ARCH["inter"])),
                       weights_backend="mlx-metal-gpu", projection_backend="cpu", **identity)
            print(f"{kind} L{layer} done ({done.sum()}/{len(done)})", flush=True)
            mx.clear_cache()
        # Only a complete layout is published to the server. Keep the partial
        # checkpoint as resume evidence instead of deleting files implicitly.
        save_numpy(path, cell=cells, done=done, side=math.ceil(math.sqrt(ARCH["inter"])),
                   weights_backend="mlx-metal-gpu", projection_backend="cpu", **identity)
        print(f"-> {path}", flush=True)


if __name__ == "__main__":
    main()
