# Intel Arc LLM Bench

LLM inference throughput benchmarks for **Intel Arc / XPU** GPUs using PyTorch and HuggingFace Transformers, with CPU comparison via llama.cpp.

## What it measures

| Metric | Description |
|---|---|
| **TTFT** | Time To First Token — latency from request to first emitted token (includes prefill) |
| **Prefill t/s** | Throughput during the prompt processing phase (compute-bound) |
| **Decode t/s** | Throughput during autoregressive generation — the main UX metric for chat |
| **Overall t/s** | End-to-end throughput: `(prompt_tokens + new_tokens) / total_time` |

## Requirements

- Linux x86_64 (tested on Arch Linux / WSL2)
- Python 3.13
- Intel Arc GPU with XPU drivers (`intel-compute-runtime`, `ocl-icd`)
- [uv](https://github.com/astral-sh/uv) package manager

## Setup

```bash
# Install all dependencies (torch+xpu, transformers, triton-xpu, accelerate)
uv sync

# Verify XPU is available
.venv/bin/python -c "import torch; print(torch.__version__, torch.xpu.is_available())"
# → 2.12.0+xpu  True

# Optional: llama.cpp GGUF benchmarks on the Arc GPU (SYCL build, needs Intel oneAPI)
make setup-llama
```

## Usage

### XPU benchmarks (HF Transformers + PyTorch XPU)

```bash
# Single model benchmark (default: Qwen2.5-0.5B-Instruct, bfloat16, 128 tokens, 3 runs)
make bench

# Quick single run
make bench-quick

# Full sweep (3 models × bf16/fp16 × 128/512 tokens) → results/bench_results.md + .csv
make sweep

# Quick sweep smoke test
make sweep-quick

# Tokenizer efficiency benchmark
make tokenizer

# Smoke tests
make test-gpu    # XPU availability
make test-hf     # HuggingFace connectivity
make test-chat   # end-to-end chat generation

# Check HuggingFace model access
make check
```

### llama.cpp benchmarks (GGUF Q4_K_M)

Runs on the Arc GPU via a SYCL build (`make setup-llama`, needs Intel oneAPI).

```bash
make sweep-llama        # full sweep on the Arc GPU → results/llama_results.md + .csv
make sweep-llama-quick  # smoke test (1 model)
make bench-llama        # single model
```

### Serve a model to opencode (OpenAI-compatible API)

`make serve-llama` exposes the GGUF model on the Arc GPU via an
OpenAI-compatible endpoint so the [opencode](https://opencode.ai) IDE (or any
OpenAI client) can use it as a local provider. It runs the **upstream
`llama-server`** binary (built with SYCL) with `--jinja`, so tool-calling is
parsed by llama.cpp's built-in, grammar-constrained parser (maintained upstream).

```bash
make setup-llama-server                             # one-off: clone + build llama-server (SYCL, needs oneAPI)
make serve-llama                                    # interactive model picker
make serve-llama SERVE_MODEL=Qwen/Qwen2.5-3B-Instruct # skip menu (fixed model)
make serve-llama SERVE_HOST=0.0.0.0 SERVE_PORT=9000  # change address/port
make serve-llama SERVE_NCTX=16384                    # fixed context (override auto)
```

With no `SERVE_MODEL`, it **prompts interactively** for which model to serve,
listing only those that fit in available memory (estimated weights + KV-cache;
the iGPU shares system RAM). The context window is set **automatically** from
the model's trained context (read from the GGUF metadata), capped at
`SERVE_MAX_CTX` (default 32768) and the model's own iGPU limit. This avoids
*"Requested tokens exceeded context window"* with large prompts/tools, and stays
within the Arc iGPU's SYCL single-allocation limit (the 7B/8B fail to create
their context at 32768, so they cap at 16384; ≤3B models run the full 32768).
Override with `SERVE_NCTX=<n>` (fixed) or `SERVE_MAX_CTX=<n>` (auto cap). The
server runs with `--parallel 1` so a single opencode client gets the full
context window (otherwise the context is split across auto slots).

**Recommended model — `Qwen/Qwen2.5-3B-Instruct` (general).** llama.cpp's native
parser (`peg-native`) only extracts `<tool_call>` tags and has **no** markdown-fence
fallback. The general 3B emits `<tool_call>` natively and works end-to-end
(verified: tool call → tool result → final answer). The **Qwen2.5-Coder** models
wrap their tool calls in ```` ```json ```` fences (a code-model bias) which the
native parser does not recover, so tool-calling breaks with them — the launcher
prints a warning if you pick one. Use a general Qwen2.5 model for opencode.

Endpoint: `http://127.0.0.1:8080/v1` (the API key is not validated). See
[SETUP.md](SETUP.md#2-llamacpp--gpu-sycl) for the `opencode.json` provider config.


### Override variables

```bash
make bench MODEL=meta-llama/Llama-3.2-1B-Instruct DTYPE=float16 NEW_TOKENS=256 RUNS=5
```

| Variable | Default | Description |
|---|---|---|
| `MODEL` | `Qwen/Qwen2.5-0.5B-Instruct` | HuggingFace model ID |
| `DTYPE` | `bfloat16` | `bfloat16` or `float16` (bf16 recommended on Arc) |
| `NEW_TOKENS` | `128` | Tokens to generate |
| `RUNS` | `3` | Benchmark runs (averaged) |
| `WARMUP` | `1` | Warmup runs (discarded) |
| `DEVICE` | `xpu` | PyTorch device |
| `LLAMA_QUANT` | `q4_k_m` | GGUF quantisation |

## Project structure

```
benchmarks/
  benchmark_tps.py        # XPU single-model TPS benchmark
  bench_sweep.py          # XPU sweep: models × dtypes × token counts
  benchmark_llama.py      # CPU single-model benchmark (llama.cpp)
  bench_llama_sweep.py    # CPU sweep (llama.cpp)
  benchmark_tokenizer.py  # tokenizer efficiency benchmark
tests/
  test_gpu.py             # XPU smoke test
  test_hf.py              # HuggingFace connectivity smoke test
  test_chat.py            # end-to-end chat generation smoke test
utils/
  check_access.py         # HuggingFace model access checker
results/
  bench_results.csv/md    # XPU sweep results
  llama_results.csv/md    # CPU (llama.cpp) sweep results
  tokenizer_results.md    # tokenizer benchmark results
```

## Results (Intel Arc iGPU, Xe2 128EU, WSL2)

### XPU — HF Transformers bf16

| Model | new_tok | Decode t/s | Prefill t/s | TTFT (ms) |
|---|---:|---:|---:|---:|
| Qwen2.5-0.5B-Instruct | 128 | **24.1** | 1665 | 51 |
| Qwen2.5-1.5B-Instruct | 128 | **12.5** | 676 | 126 |
| Llama-3.2-1B-Instruct | 128 | **16.7** | 924 | 99 |

### CPU — llama.cpp Q4_K_M

| Model | Decode t/s | Prefill t/s | GGUF MB |
|---|---:|---:|---:|
| Qwen2.5-0.5B | **19.0** | 927 | 469 |
| Qwen2.5-1.5B | **5.2** | 273 | 1408 |
| Llama-3.2-1B | **6.1** | 300 | 1277 |

> XPU wins 1.3× to 2.7× over CPU depending on model. See [results/bench_results.md](results/bench_results.md) for the full sweep.

## Documentation

- [SETUP.md](SETUP.md) — installation details and findings on failed approaches (llama.cpp GPU, IPEX-LLM)
- [XPU_STACK.md](XPU_STACK.md) — how the XPU stack works, memory/storage sizing formulas, throughput estimation

## Notes

- `bfloat16` is the recommended dtype on Intel Arc (Xe2 XMX engines).
- llama.cpp runs on the Arc GPU via a SYCL build (`make setup-llama`, needs
  Intel oneAPI).
- IPEX-LLM (INT4 quantisation on XPU) requires `transformers ~4.47`; incompatible with current `5.9.0`.
- Gated models (e.g. Llama) require a HuggingFace token: `huggingface-cli login`.
- Results are saved to `results/` and overwritten on each sweep run.
