# llama-server — configuration and log reading

Guide to the `llama-server` (the official llama.cpp binary, built with SYCL) used
to serve GGUF models to **opencode** on the Intel Arc GPU. Start it with
`make serve-llama`; the launcher is `benchmarks/serve_llama_native.py`.

Contents:
- [1. How to start](#1-how-to-start)
- [2. Configuration](#2-configuration)
- [3. Variables and flags](#3-variables-and-flags)
- [4. Reading the logs](#4-reading-the-logs)
- [5. Performance metrics (prefill vs decode)](#5-performance-metrics-prefill-vs-decode)
- [6. Hardware comparison (this iGPU vs Apple Silicon)](#6-hardware-comparison-this-igpu-vs-apple-silicon)
- [7. Common warnings (and whether they matter)](#7-common-warnings-and-whether-they-matter)
- [8. Troubleshooting](#8-troubleshooting)

---

## 1. How to start

```bash
make setup-llama-server                              # once: clone + build llama-server (SYCL, needs oneAPI)
make serve-llama                                     # interactive model picker
make serve-llama SERVE_MODEL=Qwen/Qwen2.5-3B-Instruct # fixed model (skip the menu)
make opencode-config > ~/.config/opencode/opencode.json  # generate the opencode config
```

The server exposes an OpenAI-compatible API at `http://127.0.0.1:8080/v1`
(`/v1/chat/completions`, `/v1/models`, …). The `apiKey` is not validated.

**Recommended model: `Qwen/Qwen2.5-3B-Instruct` (general).** The native
tool-calling parser (`peg-native`) only extracts `<tool_call>` tags; the
**Qwen2.5-Coder** models wrap their calls in ```` ```json ```` fences which the
parser does not recover, so tool-calling fails with them.

---

## 2. Configuration

`make serve-llama` `exec`s the binary with (see
`benchmarks/serve_llama_native.py`):

```text
llama-server \
  --model <file.gguf> \
  --alias <HF-model-id> \
  --host 127.0.0.1 --port 8080 \
  --ctx-size 32768 \
  --n-gpu-layers 999 \
  --threads 14 \
  --parallel 1 \
  --jinja
```

What each flag does:

| Flag | Default | What it does |
|---|---|---|
| `--model` | (chosen in the menu) | Path to the local GGUF (resolved/downloaded by the launcher). |
| `--alias` | HF model id | Name exposed by `GET /v1/models` — must match the `id` in opencode. |
| `--host` / `--port` | `127.0.0.1` / `8080` | Listen address. |
| `--ctx-size` | `32768` (auto) | Context window (tokens). Auto = the GGUF's training context, capped by `SERVE_MAX_CTX` and the model's own iGPU ceiling. |
| `--n-gpu-layers` | `999` | Layers on the GPU. `999` = all (full offload); `0` = CPU. |
| `--threads` | CPU core count | CPU threads (prefill / auxiliary sampling). |
| `--parallel` | `1` | Inference slots. **`1` matters:** a single client (opencode) gets the whole context window. With `N>1`, `--ctx-size` is split across the N slots. |
| `--jinja` | on | Enables native tool-calling via the GGUF's chat template. `--no-jinja` disables it. |
| `--chat-template-file` | — | Override the chat template (a Jinja file). |

---

## 3. Variables and flags

`make` variables (override on the command line, e.g.
`make serve-llama SERVE_PORT=9000`):

| Variable | Default | Effect |
|---|---|---|
| `SERVE_MODEL` | (empty) | Model to serve. Empty = interactive menu (only those that fit in memory). |
| `SERVE_HOST` | `127.0.0.1` | Listen address (`0.0.0.0` to expose on the network). |
| `SERVE_PORT` | `8080` | Port. |
| `SERVE_NCTX` | `0` | Fixed context. `0` = automatic. |
| `SERVE_MAX_CTX` | `32768` | Ceiling for the automatic mode. |
| `LLAMA_QUANT` | `q4_k_m` | GGUF quantization pattern. |
| `LLAMA_CPP_DIR` | `~/.cache/llama/llama.cpp` | `llama-server` checkout/build. |

Useful launcher flags (`serve_llama_native.py --help`): `--n-gpu-layers`,
`--n-threads`, `--parallel`, `--no-jinja`, `--chat-template-file`,
`--server-bin`, `--print-opencode-config`.

Relevant environment variables:
- `SYCL_CACHE_PERSISTENT=0` — **required** on this iGPU (Arc 140T / Xe-LPG): with
  `=1` the SYCL backend segfaults. `make` already exports it.
- `ZES_ENABLE_SYSMAN=1` — optional; lets the driver report free VRAM (silences the
  `ext_intel_free_memory is not supported` warning).
- `-lv N` (`llama-server` CLI) — log verbosity (the default 3 is very chatty; `1`
  is cleaner).

---

## 4. Reading the logs

Timestamps use the format `H.MM.SSS.mmm` (time since process start). The log has
three phases.

### Phase A — launcher (Python, before the server)

```text
== serve-llama: choose model ==
Available memory : 26.1 GiB  (safe budget 23.5 GiB)
  5  Qwen/Qwen2.5-3B-Instruct ★      2007 MB   1152 MB   3671 MB  yes   general, ctx 32768 (recommended for opencode)
...
== Native llama.cpp server (llama-server / SYCL / Intel Arc GPU) ==
Model   : Qwen/Qwen2.5-3B-Instruct  (quant=Q4_K_M)
Context : 32768 (auto: train=32768)
```

- **Memory / safe budget:** available RAM (the UMA iGPU shares system RAM) and the
  margin used to filter models.
- **Weights / +KV / Total:** GGUF size + estimated KV-cache for the context + total.
- **Context: 32768 (auto: train=32768):** the effective resolved window.

### Phase B — `llama-server` startup

```text
0.00.191.154 I   - SYCL0   : Intel(R) Graphics [0x7d51] (20770 MiB, 20770 MiB free)
0.00.191.244 I system_info: n_threads = 14 ... | AVX2 = 1 | FMA = 1 | OPENMP = 1 | REPACK = 1 |
0.00.193.074 I srv  load_model: loading model '.../qwen2.5-3b-instruct-q4_k_m.gguf'
0.01.926.365 I common_init_from_params: warming up the model with an empty run ...
0.02.395.036 I srv  load_model: initializing slots, n_slots = 1
0.02.553.568 I slot load_model: id 0 | task -1 | new slot, n_ctx = 32768
0.02.563.579 I init: chat template, example_format: '<|im_start|>system ...'
0.02.568.305 I srv  llama_server: model loaded
0.02.568.314 I srv  llama_server: server is listening on http://127.0.0.1:8080
0.02.568.329 I srv  update_slots: all slots are idle
```

- **`device_info` / `SYCL0`:** detected GPU and VRAM. Confirms it runs on the GPU.
- **`system_info`:** threads + compiled SIMD features.
- **`warming up`:** compiles kernels and allocates buffers (one empty run).
- **`n_slots = 1`, `new slot, n_ctx = 32768`:** effect of `--parallel 1` — a single
  slot with the whole window. (With `--parallel 4` you'd see 4 slots splitting the
  context.)
- **`chat template, example_format`:** detected template (Qwen ChatML
  `<|im_start|>…`). `thinking = 0` = no reasoning mode.
- **`model loaded` → `server is listening` → `all slots are idle`:** **ready.**

### Phase C — serving requests

Each request is a **task** on slot 0.

```text
0.28.884.955 I srv  params_from_: Chat format: peg-native
0.28.886.134 I slot launch_slot_: id 0 | task 0 | processing task
0.33.380.072 I slot print_timing: id 0 | task 0 | prompt eval time = 3884.99 ms / 533 tokens ( 7.29 ms per token, 137.19 tokens per second)
0.33.380.254 I slot print_timing: id 0 | task 0 |        eval time =  608.83 ms /   3 tokens (202.94 ms per token,   4.93 tokens per second)
0.33.380.256 I slot print_timing: id 0 | task 0 |       total time = 4493.82 ms / 536 tokens
0.33.380.262 I slot print_timing: id 0 | task 0 |    graphs reused = 2
0.33.380.415 I slot release: id 0 | task 0 | stop processing: n_tokens = 535, truncated = 0
```

- **`Chat format: peg-native`:** the tool-call parser in use (only reads
  `<tool_call>` tags).
- **`prompt eval time`:** **prefill** — see §5.
- **`eval time`:** **decode** — see §5.
- **`total time`:** prefill + decode.
- **`graphs reused`:** compute graphs reused from cache (lower latency).
- **`truncated = 0`:** the context fit in the window; **nothing was cut** (critical
  for opencode's tool-calling flows — `1` would mean lost tokens).

#### Large prompts and cache reuse

```text
0.38.475.612 I ... task 2 | prompt processing, n_tokens = 2048, progress = 0.26, t = 5.03 s / 407.47 tokens per second
0.48.322.210 I ... task 2 | prompt processing, n_tokens = 4096, progress = 0.51, t = 14.87 s / 275.40 tokens per second
1.19.493.884 I ... task 2 | prompt processing, n_tokens = 7951, progress = 1.00, t = 46.04 s / 172.68 tokens per second
1.23.493.390 I ... task 2 | prompt eval time = 49452.16 ms / 7960 tokens ( 6.21 ms per token, 160.96 tokens per second)
...
2.04.343.709 I slot get_availabl: id 0 | task 20 | selected slot by LCP similarity, sim_best = 0.998 (> 0.100 thold)
2.05.205.120 I ... task 20 | prompt eval time = 788.27 ms / 15 tokens ( 52.55 ms per token, 19.03 tokens per second)
```

- **`prompt processing ... progress`:** prefill progress for a large prompt; the
  rate **drops** as the context grows (attention costs more with more tokens).
- **`selected slot by LCP similarity, sim_best = 0.998`:** the new request shares
  ~99.8% of its prefix with one already cached → **the prefill is reused** and only
  the few new tokens (15) are processed. That's why `prompt eval` becomes tiny on
  follow-up requests within the same conversation.

---

## 5. Performance metrics (prefill vs decode)

`llama-server` reports **two** distinct rates — don't conflate them:

| Log metric | Operation | How it works | Ballpark (Arc 140T, 3B Q4) |
|---|---|---|---|
| `prompt eval time` / `prompt processing` | **Prefill** (process the input) | All prompt tokens in one parallel pass (compute/GEMM-bound) | **>100 t/s** on large prompts |
| `eval time` / `tg = N t/s` | **Decode** (generate the output) | 1 token per step, autoregressive (memory-bandwidth-bound) | **~13–17 t/s** |

**How to read it:**
- The speed at which text "appears" in opencode is the **decode** (`eval time`),
  ~15 t/s here.
- "tokens per second" values **above ~35** in the log are almost always
  **prefill**, not generation. That's normal and good (prefill is fast).
- Decode **slows with context size** (more KV-cache to read per token). That's why
  short-prompt measurements (~35 t/s) are higher than a long opencode conversation
  (~15 t/s).
- On very small prompts the per-token `prompt eval` looks **slow** (e.g. `19 t/s`
  for 15 tokens) — that's the fixed per-call cost dominating the average, not a
  problem.

---

## 6. Hardware comparison (this iGPU vs Apple Silicon)

Decode ("typing speed") is **memory-bandwidth-bound**, so it scales roughly with a
machine's memory bandwidth. The table compares this repo's Intel Arc iGPU with the
Mac Mini line for serving Qwen2.5 **Instruct** models to opencode (the `peg-native`
tool-calling caveat is model-specific, not hardware — always prefer a *general*
Qwen2.5, not Coder).

| Machine | Memory bandwidth | 7B Q4 decode | Best model for opencode |
|---|---|---|---|
| **Arc 140T** (this repo's iGPU, Xe-LPG) | ~120 GB/s | n/a (runs 3B at ~15 t/s) | Qwen2.5-3B-Instruct |
| **Mac Mini M4** (base) | 120 GB/s | ~20–25 t/s | Qwen2.5-7B-Instruct |
| **Mac Mini M5** (base, projected) | ~153 GB/s | ~26–32 t/s | Qwen2.5-7B-Instruct |
| **Mac Mini M4 Pro** | 273 GB/s | ~45–55 t/s | Qwen2.5-14B-Instruct |
| **Mac Mini M5 Pro** (projected) | ~300+ GB/s | ~55–65 t/s | Qwen2.5-14B-Instruct |

Notes:
- Figures are **single-stream decode** estimates (Q4_K_M, 32k context,
  `--parallel 1`, a few-thousand-token prompt) ≈ theoretical bandwidth max ×
  ~60–80% efficiency; they **drop as context fills**.
- A base **M4/M5 Mini** is the step up from this iGPU that makes **7B** comfortable;
  the **Pro** tiers are what make **14B** viable for agentic/tool-calling work that
  the Arc iGPU can't fit/run well at 32k context.
- **M5's** main edge is **prefill** (its per-GPU-core Neural Accelerators boost the
  compute-bound prompt processing more than the bandwidth-bound decode).
- **Mac Mini M5 / M5 Pro are not yet released** (expected H2 2026); those rows are
  projections from the known M5 silicon, not measured.

---

## 7. Common warnings (and whether they matter)

| Log line | Matters? | Explanation |
|---|---|---|
| `get_memory_info: [warning] ext_intel_free_memory is not supported` | No | The Level-Zero driver won't report free VRAM; it uses total memory. Silence with `ZES_ENABLE_SYSMAN=1`. |
| `control-looking token: 128247 '</s>' was not control-type ... overridden` | No | GGUF metadata quirk; llama.cpp fixes the token type itself. |
| `common_speculative_init: no implementations specified for speculative decoding` | No | Not using speculative decoding; informational only. |
| `--cache-idle-slots requires --kv-unified, disabling` | No | An optional optimization is off; no functional impact. |
| `verbosity = 3` | — | Very chatty log; lower with `-lv 1`. |

---

## 8. Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| **`SIGSEGV` / crash at startup or during inference** | `SYCL_CACHE_PERSISTENT` ended up `1`. `make` exports `=0` after `setvars.sh`; make sure it wasn't reset. |
| **`Failed to create llama_context`** | Context too large for the iGPU (7B/8B at 32768). Use a ≤3B model or lower it with `SERVE_NCTX=16384`. |
| **`truncated = 1` in the logs** | Prompt + generation exceeded `--ctx-size`. Increase the context (if it fits) or reduce the history the client sends. |
| **opencode doesn't recognize tool calls** | You're serving a **Coder** model (tool calls in ```` ```json ```` aren't extracted). Use `Qwen/Qwen2.5-3B-Instruct`. |
| **Slow decode (<10 t/s)** | Context very full, or some layers running on CPU. Check `SYCL0` in `device_info` and `--n-gpu-layers 999`. |
| **`llama-server binary not found`** | Run `make setup-llama-server` (or point `--server-bin` at it). |
| **Annoying `ext_intel_free_memory` warning** | Cosmetic; export `ZES_ENABLE_SYSMAN=1` to silence it. |

---

See also: [SETUP.md](SETUP.md) (installation and opencode config),
[SIZING.md](SIZING.md) (memory/context), [README.md](README.md).
