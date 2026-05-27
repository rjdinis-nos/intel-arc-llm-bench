# Intel Arc LLM Bench

LLM inference throughput benchmarks for **Intel Arc / XPU** GPUs using PyTorch and HuggingFace Transformers.

## What it measures

| Metric | Description |
|---|---|
| **TTFT** | Time To First Token — latency from request to first emitted token (includes prefill) |
| **Prefill t/s** | Throughput during the prompt processing phase (compute-bound) |
| **Decode t/s** | Throughput during autoregressive generation — the main UX metric for chat |
| **Overall t/s** | End-to-end throughput: `(prompt_tokens + new_tokens) / total_time` |

## Requirements

- Linux x86_64
- Python 3.13
- Intel Arc GPU with XPU drivers
- [uv](https://github.com/astral-sh/uv) package manager

## Setup

```bash
uv sync
```

This installs PyTorch XPU, `transformers`, `accelerate`, and `triton-xpu` from the Intel/PyTorch XPU index automatically.

## Usage

```bash
# Single model benchmark (default: Qwen2.5-0.5B-Instruct, bfloat16, 128 tokens, 3 runs)
make bench

# Quick single run
make bench-quick

# Full sweep (models × dtypes × token counts) → results/bench_results.md + .csv
make sweep

# Quick sweep smoke test
make sweep-quick

# Tokenizer efficiency benchmark (encode/decode speed, chars/token by language)
make tokenizer

# Smoke tests
make test-gpu    # XPU availability
make test-hf     # HuggingFace connectivity
make test-chat   # end-to-end chat generation

# Check HuggingFace model access (gated models)
make check
```

### Override variables

```bash
make bench MODEL=meta-llama/Llama-3.2-1B-Instruct DTYPE=float16 NEW_TOKENS=256 RUNS=5
```

| Variable | Default |
|---|---|
| `MODEL` | `Qwen/Qwen2.5-0.5B-Instruct` |
| `DTYPE` | `bfloat16` |
| `NEW_TOKENS` | `128` |
| `RUNS` | `3` |
| `WARMUP` | `1` |
| `DEVICE` | `xpu` |

## Project structure

```
benchmarks/
  benchmark_tps.py        # single-model TPS benchmark (prefill + decode + TTFT)
  bench_sweep.py          # sweep over models × dtypes × token counts
  benchmark_tokenizer.py  # tokenizer encode/decode speed and token efficiency
tests/
  test_gpu.py             # XPU smoke test
  test_hf.py              # HuggingFace connectivity smoke test
  test_chat.py            # end-to-end chat generation smoke test
utils/
  check_access.py         # HuggingFace model access checker
results/
  bench_results.csv       # raw sweep results
  bench_results.md        # formatted markdown table
  tokenizer_results.md    # tokenizer benchmark results
```

## Sample results

| Model | dtype | new_tok | TTFT (ms) | Prefill t/s | Decode t/s | Overall t/s |
|---|---|---:|---:|---:|---:|---:|
| `Qwen2.5-0.5B-Instruct` | bfloat16 | 128 | 60.5 | 1405 | 24.80 | 24.71 |

> Measured on Intel Arc (XPU). See [results/bench_results.md](results/bench_results.md) for full sweep results.

## Notes

- `bfloat16` is the recommended dtype on Intel Arc (Xe2 / XMX engines).
- Gated models (e.g. Llama) require a HuggingFace token: `huggingface-cli login`.
- Results are saved to `results/` and overwritten on each sweep run.
