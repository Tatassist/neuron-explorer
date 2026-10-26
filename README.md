# Neuron Explorer — MLX / LFM2.5

A fork of [lyramakesmusic/neuron-explorer](https://github.com/lyramakesmusic/neuron-explorer), adapted for **LFM2.5-350M on Apple Silicon with MLX/Metal**.

The default checkpoint is the already-installed `LiquidAI/LFM2.5-350M-MLX-8bit` under `$HOME/.lmstudio/models`. It uses 8-bit affine weights, group size 64. A `bfloat16` entry in its config does not mean the weights are unquantized.

## Run

```bash
bash scripts/run_mlx.sh
# http://127.0.0.1:8791
```

The launcher reuses `$HOME/venvs/mlx/bin/python3` when available. To select another existing environment:

```bash
EXPLORER_PYTHON=/path/to/python bash scripts/run_mlx.sh
```

Otherwise it uses `uv run` with the checked-in MLX dependencies. The launcher enables offline model loading. No weights are downloaded; no CPU or hosted inference fallback is enabled. Run GPU commands with normal macOS Metal access, outside a restricted sandbox.

`EXPLORER_MODEL` can select another local LFM2/LFM2.5 MLX checkpoint. Model config, tensor dimensions and weight-file identities determine an isolated profile under `data/`. Bundled Qwen caches, neuron names and experiments are never used for LFM.

## Available

1. **3D map:** 16 hybrid convolution/attention layers, 4,608 SwiGLU neurons per layer, hidden width 1,024. The MLP width comes from the actual tensors; LFM adjusts the config's nominal feed-forward width. Convolution layers are labelled as convolution and have no attention heads.
2. **Activation inspection:** paste text into `run` or `fires`, select a token, and inspect positive/negative activations. Neurons can be opened directly with a jump such as `L15 N100`.
3. **Steered chat:** compare positive, baseline and negative interventions using the existing chat panel. Each row has its own conversation and hybrid convolution/KV cache. Rows run sequentially on MLX. Additive doses are fractions of the layer's median reference residual norm, divided by the down-column norm. Multiplicative mode retains upstream semantics: x dose / x 1 / x minus-dose (including a sign flip in the negative row).
4. **Weight analyses:** per-neuron positive/negative logit lenses, token search after index building, and direct connections computed with LFM's RMSNorm convention. These are weight-only hypotheses, not proof of circuits.
5. **Research API/MCP:** activation reads, multi-neuron interventions, attention-head ablation on attention layers, next-token logprobs and comparisons. The optional upstream Claude tab still requires its CLI and login; starting the explorer does not launch a Claude conversation.

At startup, activation ranges are measured over a **small local reference text**, with source text and token count saved in the profile. Scores are ratios to this calibration; they are not corpus percentiles or broad claims about a neuron's normal range.

## Build token search data

The map and a selected neuron's lenses work without a full index. To enable token search over every neuron:

```bash
$HOME/venvs/mlx/bin/python3 scripts/build_mlx_lens.py
# Or: uv run python scripts/build_mlx_lens.py
```

This is an explicit weight-only indexing job, not a generation sweep. Its output stays under the current model profile. The server discovers the completed index without a restart.

## Focused runtime check

```bash
$HOME/venvs/mlx/bin/python3 scripts/check_mlx.py
```

Checks unmodified-versus-instrumented logits, the exact output delta from an additive neuron intervention, zero-dose parity, cleanup of intervention state, activation dimensions, and three independent zero-dose generations. The local report is `work/mlx-check.json` (ignored by Git).

## Remaining upstream features

The CUDA/vLLM sweep scripts, rollout embeddings, weight-sorted layouts and precomputed multi-hop connection indexes have not been ported to MLX. LFM initially has no sweep rollouts or named concepts. The explorer shows direct connections for a selected neuron; token-to-token network overlays require a precomputed connection index. It does not substitute Qwen experiments for missing LFM data.

The original Qwen/PyTorch route remains explicit:

```bash
EXPLORER_BACKEND=torch uv run --extra qwen python explorer/server.py
```

See [README.upstream.md](README.upstream.md) and the unchanged upstream research log for the original Qwen experiments. Their neuron identities and findings do not transfer to LFM.
