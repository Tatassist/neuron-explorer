"""Dispatch to MLX by default; the original CUDA route requires backend torch."""
import os

if __name__ == "__main__":
    if os.environ.get("EXPLORER_BACKEND", "mlx") == "mlx":
        from neuron_stats_mlx import main
        main()
    else:
        import runpy
        from pathlib import Path
        runpy.run_path(str(Path(__file__).with_name("neuron_stats_torch.py")), run_name="__main__")
