# agentorskill-cli

Evaluate migration libraries (TorchAX, MindTorch, …) with a reproducible CLI: hardware probe, optional LLM-based README extraction → `AdapterSpec`, then **smoke / core / models / training / numeric / benchmark** suites, **adaptation-rate summaries**, **numeric fidelity vs CPU baseline**, and JSON + Markdown reports.

## Install

```bash
cd /path/to/-demo
pip install -e .
```

For local test development, install the test extra:

```bash
pip install -e ".[test]"
pytest -q
```

The package intentionally does not install PyTorch, MindSpore, JAX, CANN, or vendor migration libraries by default. Install those in the target evaluation environment as needed.

## LLM configuration (API key, model, base URL)

Use these only when you call `extract` or `run` **without** `--adapter-file`, so the tool can read README/docs and produce an `AdapterSpec`. Runs with `--adapter-file` use the hand-written adapter directly and do not call an LLM.

### Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | **Yes** for LLM extraction | API key for an OpenAI-compatible Chat Completions endpoint. |
| `OPENAI_MODEL` | No | Model name. Default in code: `gpt-4o-mini` if unset. |
| `OPENAI_BASE_URL` | No | Override API base URL (official OpenAI default is used if unset). |

If you use Azure OpenAI, a reverse proxy, or another vendor that exposes an OpenAI-compatible API, set `OPENAI_BASE_URL` to the URL they document (check whether the path must include `/v1`).

### CLI overrides

The same settings can be passed per command (they override env vars where applicable):

- `--model` — model name (also reads `OPENAI_MODEL`).
- `--openai-base-url` — base URL (also reads `OPENAI_BASE_URL`).

Example (PowerShell):

```powershell
$env:OPENAI_API_KEY = "sk-..."
$env:OPENAI_MODEL = "gpt-4o-mini"
$env:OPENAI_BASE_URL = "https://api.openai.com/v1"   # optional; omit for default

eval-migration extract --local-path path\to\repo --out adapter.json
eval-migration run --local-path path\to\repo --suite smoke --yes
```

Example with explicit URL and model (no env for model/base):

```powershell
$env:OPENAI_API_KEY = "sk-..."

eval-migration run `
  --local-path path\to\repo `
  --model gpt-4o-mini `
  --openai-base-url "https://your-compatible-endpoint.example/v1" `
  --suite smoke --yes
```

**Note:** With `--adapter-file`, no LLM call is made; API key and URL are not needed.

## Suites

| Suite | Role |
|-------|------|
| `smoke` | Minimal import / tensor / matmul sanity checks. |
| `core` | Broad operator coverage: tensor shape ops, math/reductions, NN layers, losses + single optimizer steps, plus legacy conv/LayerNorm/dropout/autograd samples. |
| `models` | Small MLP, CNN, single Transformer encoder layer (forward). |
| `training` | Multi-step SGD / AdamW training loops, grad clip / train-eval, optional CUDA AMP (skipped without CUDA). |
| `numeric` | Same MLP twice (baseline **without** adapter preamble vs **with** preamble); compares **full** flattened outputs and gradients with `torch.allclose`-style rules (`--numeric-rtol`, `--numeric-atol`). |
| `benchmark` | Rough timing micro-benchmarks. |
| `all` | Runs in order: smoke → core → models → training → numeric → benchmark. If smoke fails, benchmark is skipped. |

## Adaptation rate and reports

Each case has a **category** (e.g. `tensor`, `math`, `nn`, `training`, `numeric_fidelity`, `perf`). After a run, JSON/Markdown include **`evaluation_summary`**:

- **`adaptation_rate_overall`**: `passed / total` over all **counted** cases in the selected **adaptation suites** (skips are excluded from the denominator).
- **`by_category` / `by_suite`**: same rate split by category or suite.
- **Default suites in the rate**: `core`, `models`, `training`, `numeric` (not `smoke` or `benchmark` unless you opt in).

CLI:

- `--adaptation-suites core,models,training,numeric` — override which suites contribute to the rate (comma-separated).
- `--include-smoke-in-adaptation` — also include the `smoke` suite in the rate.
- `--honor-adapter-device` — after the adapter preamble, call `torch.set_default_device(...)` using `AdapterSpec.device.target_device` (e.g. `jax` for TorchAX) so ops run on that backend where PyTorch supports it.

Numeric fidelity fields (`numeric_fidelity` in the report) include `within_tolerance_y`, `within_tolerance_g`, `max_abs_err_*`, `rmse_*`.

## Usage

```bash
# Show help
eval-migration --help

# Probe hardware (with confirmation)
eval-migration probe

# Run with a hand-written adapter JSON (no LLM)
eval-migration run --adapter-file examples/torchax_adapter.json --suite all --yes

# Full run on JAX-style device (TorchAX)
eval-migration run --adapter-file examples/torchax_adapter.json --suite all --yes --honor-adapter-device

# Extract adapter from docs + run (requires OPENAI_API_KEY)
eval-migration run --readme-url https://raw.githubusercontent.com/google/torchax/main/README.md --suite smoke --yes
```

## Adapter JSON

See `examples/` for minimal `AdapterSpec` samples.

### Skipped cases

A subtest can return `{"ok": true, "skipped": true, "reason": "..."}` (e.g. SDPA unavailable, no CUDA for AMP). Skips are **not** counted as failures and are **excluded** from the adaptation-rate denominator.
