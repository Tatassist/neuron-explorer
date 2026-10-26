#!/bin/bash
set -euo pipefail
EXPLORER_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export EXPLORER_BACKEND=mlx
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export UV_CACHE_DIR="$EXPLORER_ROOT/.cache/uv"
if [[ -n "${EXPLORER_PYTHON:-}" ]]; then
    exec "$EXPLORER_PYTHON" "$EXPLORER_ROOT/explorer/server.py"
elif [[ -x "$HOME/venvs/mlx/bin/python3" ]]; then
    exec "$HOME/venvs/mlx/bin/python3" "$EXPLORER_ROOT/explorer/server.py"
else
    cd "$EXPLORER_ROOT"
    exec uv run python explorer/server.py
fi
