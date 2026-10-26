"""Check a completed sweep's row alignment, controls, captures and layouts."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "explorer"))

import numpy as np
from analysis_utils import file_sha256, save_json
from model_profile import ARCH, CACHE, PROFILE


def check(path):
    meta = json.loads(path.with_suffix(".meta.json").read_text())
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert meta["profile"] == PROFILE and file_sha256(meta["stats"]) == meta["stats_sha256"]
    stats = np.load(meta["stats"], allow_pickle=False)
    down_norm = np.linalg.norm(stats["w_down"], axis=1)
    controls = np.load(path.with_suffix(".controls.npz"), allow_pickle=False)
    np.testing.assert_allclose(np.linalg.norm(controls["vectors"], axis=1), down_norm[controls["refs"]], rtol=2e-6)
    keys = [(r["kind"], r["neuron"], r["prompt"], r["dose"]) for r in rows]
    expected = {("baseline", -1, p, 0.0) for p in meta["prompts"]}
    expected |= {(kind, n, p, d) for kind, ns in (("neuron", meta["neurons"]), ("random", range(meta["random_controls"])))
                 for n in ns for p, d in meta["design"]}
    assert len(set(keys)) == len(keys) and set(keys) == expected
    for row in rows:
        assert row["layer"] == meta["layer"] and row["finish"] in ("stop", "length")
        assert 1 <= row["n_gen"] <= meta["max_tokens"]
        assert row["n_gen"] == len(row["token_ids"]) == len(row["logprob"]) == len(row["top_ids"]) == len(row["top_lps"])
        assert np.isfinite(row["logprob"]).all() and max(row["logprob"]) <= 1e-6
        for token, lp, ids, lps in zip(row["token_ids"], row["logprob"], row["top_ids"], row["top_lps"]):
            assert len(ids) == len(lps) == meta["top_k"] and len(set(ids)) == len(ids)
            assert token == ids[0] and abs(lp - lps[0]) < 1e-6
            assert np.isfinite(lps).all() and np.all(np.diff(lps) <= 0)
        if row["kind"] == "baseline":
            assert row["ref_neuron"] is None and row["add"] == row["rel_push"] == 0
        else:
            ref = row["ref_neuron"]
            assert ref == (row["neuron"] if row["kind"] == "neuron" else int(controls["refs"][row["neuron"]]))
            if meta["dose_unit"] == "rel":
                np.testing.assert_allclose(row["rel_push"], abs(row["dose"]), rtol=2e-6)
            else:
                np.testing.assert_allclose(row["add"], row["dose"] * float(stats["abs_max"][ref]), rtol=2e-6)
        if meta["capture_layers"]:
            captured = np.load(Path(meta["capture_root"]) / meta["capture_tag"] / f"{row['capture_id']}.npz", allow_pickle=False)
            np.testing.assert_array_equal(captured["layers"], meta["capture_layers"])
            assert captured["mlp_in"].shape == (len(meta["capture_layers"]), ARCH["hidden"])
            assert np.isfinite(captured["mlp_in"]).all() and str(captured["profile"]) == PROFILE
    embedding = np.load(path.with_suffix(".emb.npy"), allow_pickle=False)
    assert embedding.shape == (len(rows), 384) and np.isfinite(embedding).all()
    np.testing.assert_allclose(np.linalg.norm(embedding.astype(np.float32), axis=1), 1.0, atol=1e-3)
    points = json.loads(path.with_suffix(".umap.json").read_text())
    assert points["profile"] == PROFILE and points["source_sha256"] == file_sha256(path)
    assert points["row"] == list(range(len(rows))) and points["encoder"]["backend"] == "mlx-metal-gpu"
    for k in ("kind", "neuron", "prompt", "dose"):
        assert points[k] == [r[k] for r in rows]
    for k in ("x", "y"):
        assert len(points[k]) == len(rows) and np.isfinite(points[k]).all() and min(points[k]) >= 0 and max(points[k]) <= 1
    layouts = {}
    for kind in ("writes", "reads"):
        layout = np.load(CACHE / f"layout_{kind}.npz", allow_pickle=False)
        assert str(layout["profile"]) == PROFILE and layout["done"].all()
        cells = layout["cell"]
        assert cells.shape == (ARCH["n_layers"], ARCH["inter"])
        for cells_in_layer in cells:
            np.testing.assert_array_equal(np.sort(cells_in_layer), np.arange(ARCH["inter"]))
        layouts[kind] = {"shape": list(cells.shape), "side": int(layout["side"]), "unique_cells_per_layer": True}
    report = {"profile": PROFILE, "sweep": str(path.resolve()), "rows": len(rows),
              "neurons": len(meta["neurons"]), "random_controls": meta["random_controls"],
              "unique_complete_design": True, "token_logprob_alignment": True,
              "control_norms_match": True, "capture_rows_valid": True,
              "embedding_shape": list(embedding.shape), "embedding_rows_match_sweep": True, "layouts": layouts}
    save_json(Path(__file__).resolve().parents[1] / "work/mlx-artifacts-check.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("jsonl", type=Path)
    check(ap.parse_args().jsonl)
