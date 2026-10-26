"""Model identity and data locations, without importing a GPU runtime."""
import hashlib
import json
import os
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = os.environ.get("EXPLORER_BACKEND", "mlx")
NO_MODEL = os.environ.get("EXPLORER_NO_MODEL", "").lower() in ("1", "true", "yes")

if BACKEND == "mlx":
    MODEL = str(Path(os.environ.get(
        "EXPLORER_MODEL", str(Path.home() / ".lmstudio/models/LiquidAI/LFM2.5-350M-MLX-8bit")
    )).expanduser().resolve())
    model_dir = Path(MODEL)
    if not (model_dir / "config.json").is_file():
        raise FileNotFoundError(f"Local LFM model missing: {MODEL}. Set EXPLORER_MODEL to the installed MLX checkpoint.")
    config = json.loads((model_dir / "config.json").read_text())
    if config["model_type"] != "lfm2":
        raise ValueError("The MLX backend currently supports LFM2/LFM2.5 checkpoints only")
    # LFM adjusts block_ff_dim before building the MLP. Read the actual tensor width,
    # rather than treating config.intermediate_size as a neuron count.
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]
    key = "model.layers.0.feed_forward.w1.weight"
    weight_file = model_dir / index[key]
    with weight_file.open("rb") as f:
        header = json.loads(f.read(struct.unpack("<Q", f.read(8))[0]))
    quant = config.get("quantization", {})
    ARCH = {"n_layers": config["num_hidden_layers"], "hidden": config["hidden_size"],
            "inter": header[key]["shape"][0], "layer_types": config["layer_types"],
            "attn_heads": config["num_attention_heads"], "kv_heads": config["num_key_value_heads"],
            "linear_k_heads": 0, "linear_v_heads": 0, "conv_kernel": config["conv_L_cache"]}
    identity = hashlib.sha256((model_dir / "config.json").read_bytes())
    for name in sorted(set(index.values())):
        stat = (model_dir / name).stat()
        identity.update(f"{MODEL}/{name}:{stat.st_size}:{stat.st_mtime_ns}".encode())
    PROFILE = f"lfm2-mlx-q{quant.get('bits', 'none')}-{identity.hexdigest()[:12]}"
    MODEL_NAME = model_dir.name
    DATA = ROOT / "data" / PROFILE
    CACHE = DATA / "cache"
    RUNS, STATS, NAMES = DATA / "runs", DATA / "stats", DATA / "names.json"
elif BACKEND == "torch":
    from transformers import AutoConfig
    MODEL = os.environ.get("EXPLORER_MODEL", "Qwen/Qwen3.5-4B")
    cfg = AutoConfig.from_pretrained(MODEL).get_text_config()
    ARCH = {"n_layers": cfg.num_hidden_layers, "hidden": cfg.hidden_size, "inter": cfg.intermediate_size,
            "layer_types": list(cfg.layer_types), "attn_heads": cfg.num_attention_heads,
            "kv_heads": cfg.num_key_value_heads, "linear_k_heads": cfg.linear_num_key_heads,
            "linear_v_heads": cfg.linear_num_value_heads}
    MODEL_NAME, PROFILE = MODEL, "qwen35-torch"
    DATA, CACHE = ROOT, ROOT / "explorer" / "cache"
    RUNS, STATS, NAMES = ROOT / "runs", ROOT / "stats", ROOT / "explorer" / "names.json"
else:
    raise ValueError("EXPLORER_BACKEND must be mlx or torch")
