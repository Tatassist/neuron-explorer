# Neuron Explorer — MLX / LFM2.5

A fork of [lyramakesmusic/neuron-explorer](https://github.com/lyramakesmusic/neuron-explorer), adapted for **LFM2.5-350M on Apple Silicon with MLX/Metal**.

The default checkpoint is the already-installed `LiquidAI/LFM2.5-350M-MLX-8bit` under `$HOME/.lmstudio/models`. It uses 8-bit affine weights, group size 64. A `bfloat16` entry in its config does not mean the weights are unquantized.

## Run

```bash
bash scripts/run_mlx.sh
# http://127.0.0.1:8791
```

The launcher uses the project's `.venv` when present, then the existing `$HOME/venvs/mlx/bin/python3`. To select another existing environment:

```bash
EXPLORER_PYTHON=/path/to/python bash scripts/run_mlx.sh
```

Otherwise it uses `uv run` with the checked-in MLX dependencies. Python 3.12 is selected for project setup; Python 3.11+ and NumPy 2.3.3+ avoid the old Accelerate/SME numerical warnings on M4. The launcher enables offline model loading. No weights are downloaded; no CPU or hosted inference fallback is enabled. Run GPU commands with normal macOS Metal access, outside a restricted sandbox.

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

## MLX sweeps, response embeddings and weight layouts

Run these from the checkout, with normal Metal access. Install the analysis dependencies into the project environment first (the shared MLX environment is unchanged):

```bash
UV_CACHE_DIR="$PWD/.cache/uv" uv sync --extra analysis --python 3.12
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export NUMBA_CACHE_DIR="$PWD/.cache/numba"
```

1. **Calibrate a layer.** The default is a small 96-token local reference, measuring 95 positions after excluding the first token. For substantive experiments, pass your own representative UTF-8 corpus with `--text-file /path/to/corpus.txt --tokens 100000`. Text is never silently repeated or downloaded. `--layers all` calibrates all layers.

   ```bash
   .venv/bin/python sweep/neuron_stats.py --layers 15
   ```

2. **Run a bounded sweep.** Resolve the current profile once, then use it in the commands below.

   ```bash
   EXPLORER_DATA="$(.venv/bin/python -c 'import sys; sys.path.insert(0,"explorer"); from model_profile import DATA; print(DATA)')"
   .venv/bin/python sweep/steer_sweep.py \
     --stats "$EXPLORER_DATA/stats/lfm_L15.npz" \
     --neurons 0-7 --random-controls 4 --max-tokens 64 \
     --run mainrel-mlx-smoke-L15
   ```

   This produces 124 rollouts: 8 neurons and 4 random controls, 10 prompt/dose combinations each, plus 4 baselines. `--neurons all`, `random:128`, and comma/range selections work. Upstream `main`, `calib`, and `main_rel` designs are preserved; use `--dose-unit absmax` with `main`/`calib` to reproduce their activation-range convention. Default `main_rel` doses use fractions of the calibration's median residual norm. All examples run sequentially with independent hybrid caches. The four-control sample is a pipeline check, not a well-estimated null distribution or a model-wide finding.

   Rerun the same command to resume. Model/profile, stats hash, seed, neuron selection, prompts, doses, token cap and capture settings must match; changed settings require a new run name. A run lock prevents concurrent writers. JSONL preserves greedy token IDs, chosen logprobs, top-k alternatives, stopping reason and the actual push size. EOS is included in token records and excluded from text. Seeded random directions and their reference neurons are saved in `<run>.controls.npz`.

   Captures go to `data/<profile>/captures/<run>/<capture_id>.npz`, with `layers` and float16 `mlp_in[capture_layer, hidden]` at the last prompt token (after `ffn_norm`, before `w1/w3`). The default relative layers are `+0,+1,+4,+8`, restricted to the model's 16 layers. `--capture-layers ''` disables captures. Completed rows flush immediately. If interrupted during a JSONL write, the script reports the damaged row rather than discarding data.

3. **Build the response map.** The cached `sentence-transformers/all-MiniLM-L6-v2` encoder runs entirely on MLX/Metal, with 256-token truncation, mean pooling and normalized 384-dimensional vectors. By default, it embeds the first 600 characters of each response. It groups responses by sentence similarity; these are not LFM hidden states. The existing local revision is `c9745ed1d9f207416be6d2e6f8de32d1f16199bf`; `--model` accepts another compatible local snapshot directory.

   ```bash
   .venv/bin/python explorer/build_embeddings.py \
     "$EXPLORER_DATA/runs/mainrel-mlx-smoke-L15.jsonl"
   ```

   Outputs `<run>.emb.npy` and `<run>.umap.json`, preserving the original JSONL row indices and model/source provenance. UMAP runs on CPU as a visualization algorithm. Runs with fewer than four points use a labelled deterministic PCA projection. The server withholds stale embeddings after a sweep grows; rebuild them when the run finishes.

4. **Build both weight layouts.** Uses the actual dequantized MLX weights: `w2` columns for **writes**, concatenated `w1/w3` rows for **reads**. Weight loading, dequantization and normalization run on Metal; randomized PCA (50 dimensions) and UMAP run on CPU. This analysis uses no model inference fallback.

   ```bash
   .venv/bin/python explorer/build_layout.py --kind both
   ```

   Produces complete `cache/layout_writes.npz` and `cache/layout_reads.npz` within the current profile. Each covers all 16 layers and 4,608 neurons, snapped into a 68-wide grid with a partial final row. Checkpoints resume by layer; only complete layouts are published. The map's **index / writes / reads** controls select them.

Refresh the explorer after creating a new run. Select layer 15, open one of its sampled neurons, and use **examples** or **embeddings**. Other neurons still have live activation reads and steering, but no recorded sweep examples until swept. Scripts default to MLX; `EXPLORER_BACKEND=torch` selects their preserved upstream implementations.

Additional focused checks:

```bash
.venv/bin/python scripts/check_mlx_analysis.py
# Optional same-checkpoint reference comparison, using an existing MPS/PyTorch environment:
$HOME/venvs/mlx/bin/python3 scripts/check_mlx_analysis.py --reference-mps
```

These check greedy decoding against MLX-LM, EOS/length records, zero-dose parity, neuron-versus-output direction equivalence, capture shape, state cleanup, MiniLM normalization and batch/padding invariance. The optional reference comparison checks the cached MiniLM's embeddings against Transformers on MPS, including Unicode, empty text and truncation. Reports stay under `work/` and model data stays under `data/`, both ignored by Git.

## Remaining upstream features

Precomputed multi-hop connection indexes have not been ported to MLX. The explorer shows direct connections for a selected neuron; token-to-token network overlays require a precomputed connection index. Neuron concepts are not automatically named, and Qwen experiments are not substituted for missing LFM data.

The original Qwen/PyTorch route remains explicit:

```bash
EXPLORER_BACKEND=torch uv run --extra qwen python explorer/server.py
```

See [README.upstream.md](README.upstream.md) and the unchanged upstream research log for the original Qwen experiments. Their neuron identities and findings do not transfer to LFM.
