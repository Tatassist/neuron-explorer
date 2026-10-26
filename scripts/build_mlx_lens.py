"""Build model-specific token search data on MLX/Metal, without running a sweep."""
import argparse
import os
import sys
from pathlib import Path

os.environ["EXPLORER_BACKEND"] = "mlx"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "explorer"))
from model_profile import CACHE, STATS
from runtime_mlx import Runtime

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunk", type=int, default=64)
    args = parser.parse_args()
    if not 1 <= args.chunk <= 512:
        parser.error("--chunk must be between 1 and 512")
    runtime = Runtime(STATS)
    print(runtime.build_lens_index(CACHE / "lens_index.npz", chunk=args.chunk))
