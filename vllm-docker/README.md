# vLLM Server (Intel XPU) with Docker Compose

OpenAI-compatible LLM API server using **vLLM on Intel Arc / XPU**, in Docker.
Built from vLLM's official `Dockerfile.xpu` (`vllm-openai` target).

> ⚠️ There is **no prebuilt XPU image** on Docker Hub — you build it once from
> source with `make build`. The `vllm/vllm-openai:latest` image on Docker Hub is
> CUDA-only and will **not** run on Intel GPUs.

## Requirements

- Docker + Docker Compose v2
- Intel Arc GPU (Xe / Xe2). On **native Linux** the GPU is `/dev/dri`; on
  **WSL2** it's `/dev/dxg` + `/usr/lib/wsl/lib` (auto-detected by the Makefile).
- ~30+ min and several GB of disk for the one-time image build.

> ⚠️ **WSL2 caveat:** Intel GPU compute inside WSL2 is unreliable (see
> [../SETUP.md](../SETUP.md) — llama.cpp GPU did not work in WSL2). If the server
> fails to initialise the XPU under WSL2, run on a native Linux host instead.

## Quick Start

### 1. Configure

```bash
cp .env.example .env
# Edit .env if needed (MODEL_NAME, DTYPE, MAX_MODEL_LEN, ...)
```

### 2. Build the XPU image (one-time)

```bash
make build            # builds vllm-xpu:v0.22.0 from vLLM source
```

### 3. Start the server

```bash
make start            # auto-selects the WSL2 or native /dev/dri GPU override
make logs             # watch startup (model download + XPU warmup take minutes)
```

### 4. Test

```bash
make health           # health check + list models
make test             # sample chat completion
```

Or with curl:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

### 5. Connect Copilot CLI

Point Copilot CLI at the OpenAI-compatible endpoint:

```
http://localhost:8000/v1/
```

## Make Targets

```bash
make build      # build the vLLM XPU image from source (one-time, slow)
make start      # start the server (auto WSL2 vs native /dev/dri)
make stop       # stop the server
make restart    # restart the server
make logs       # follow container logs
make status     # check if the server is running
make health     # health check + list models
make test       # sample chat completion
make config     # show the resolved compose configuration
make cleanup    # stop + remove volumes (deletes cached models)
```

Override variables inline, e.g.:

```bash
make build VLLM_REF=v0.22.0
make start MODEL_NAME=Qwen/Qwen2.5-1.5B-Instruct
```

## Configuration (`.env`)

| Variable | Default | Description |
|---|---|---|
| `MODEL_NAME` | `Qwen/Qwen2.5-0.5B-Instruct` | HuggingFace model ID |
| `DTYPE` | `bfloat16` | `bfloat16` (recommended on Arc), `float16`, `float32` |
| `MAX_MODEL_LEN` | `2048` | Context window (lower if OOM) |
| `VLLM_PORT` | `8000` | Host port |
| `GPU_MEMORY_UTILIZATION` | `0.9` | Fraction of GPU memory vLLM reserves |
| `TENSOR_PARALLEL_SIZE` | `1` | GPUs for tensor parallelism |
| `SHM_SIZE` | `4gb` | Container shared memory |
| `EXTRA_ARGS` | _(empty)_ | Extra bare flags for `vllm serve` |
| `VLLM_REF` | `v0.22.0` | vLLM source tag to build |
| `VLLM_IMAGE` | `vllm-xpu:v0.22.0` | Built image tag used by compose |
| `HUGGING_FACE_HUB_TOKEN` | _(empty)_ | Token for gated models (Llama, ...) |

## How GPU passthrough is wired

`make` composes the base file with one GPU override:

- **Native Linux** → `docker-compose.dri.yml` (passes `/dev/dri`, adds
  `render`/`video` groups).
- **WSL2** → `docker-compose.wsl.yml` (passes `/dev/dxg`, mounts
  `/usr/lib/wsl/lib`, sets `LD_LIBRARY_PATH`).

To run manually:

```bash
docker compose -f docker-compose.yml -f docker-compose.dri.yml up -d   # native
docker compose -f docker-compose.yml -f docker-compose.wsl.yml up -d   # WSL2
```

## Common Models

```
Qwen/Qwen2.5-0.5B-Instruct     # small, fast — good first test
Qwen/Qwen2.5-1.5B-Instruct     # medium
meta-llama/Llama-3.2-1B-Instruct   # gated → set HUGGING_FACE_HUB_TOKEN
TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

## Troubleshooting

**Build fails / very slow** — the XPU image compiles vLLM from source; expect
30+ min and a few GB. Ensure enough disk and `--shm-size` headroom.

**Server never becomes healthy** — first start downloads the model and warms up
the XPU; `start_period` is 5 min. Watch `make logs`.

**XPU not found / falls back or crashes (WSL2)** — known WSL2 limitation; try a
native Linux host. Verify on the host with the repo's `make test-gpu`.

**Out of memory** — lower `MAX_MODEL_LEN`, lower `GPU_MEMORY_UTILIZATION`
(e.g. 0.7), or use a smaller model.

**Gated model 401** — set `HUGGING_FACE_HUB_TOKEN` in `.env`.

## Cleanup

```bash
make stop                 # stop the container
make cleanup              # stop + delete cached models (volume)
docker rmi vllm-xpu:v0.22.0   # remove the built image
rm -rf .vllm-src              # remove cloned build source
```

## Links

- [vLLM Docs](https://docs.vllm.ai/)
- [vLLM on Intel XPU](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/xpu.html)
- [vLLM OpenAI-compatible server](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html)
