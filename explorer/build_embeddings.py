"""Dispatch to the selected model backend (MLX by default)."""
import os

if __name__ == "__main__":
    if os.environ.get("EXPLORER_BACKEND", "mlx") == "mlx":
        from build_embeddings_mlx import main
        main()
    else:
        import runpy
        from pathlib import Path
        runpy.run_path(str(Path(__file__).with_name("build_embeddings_torch.py")), run_name="__main__")
