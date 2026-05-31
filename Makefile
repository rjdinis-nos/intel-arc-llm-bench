# Intel XPU LLM Benchmark — Makefile
# ----------------------------------
# Uso:
#   make            # mostra ajuda
#   make bench      # benchmark único (TPS)
#   make sweep      # sweep completo (modelos × dtypes × tokens)
#   make tokenizer  # benchmark de tokenizers
#   make check      # verifica acesso aos modelos HF
#   make test-gpu   # smoke test do GPU (XPU)
#   make test-hf    # smoke test do HuggingFace
#   make test-chat  # smoke test de chat
#   make all        # check + bench + sweep + tokenizer
#   make clean      # apaga caches __pycache__ e resultados

PY := .venv/bin/python

# Source the Intel oneAPI runtime (DPC++/MKL/SYCL) for the SYCL llama.cpp build
# and at runtime. Override ONEAPI_ENV if your toolkit lives elsewhere.
# setvars.sh forces SYCL_CACHE_PERSISTENT=1, but the oneAPI 2026.0 persistent
# device-code (JIT) cache segfaults on the Arc 140T (Xe-LPG) iGPU during
# inference, so we disable it after sourcing.
ONEAPI_ENV ?= /opt/intel/oneapi/setvars.sh
SYCL_ENV   := $(if $(wildcard $(ONEAPI_ENV)),. $(ONEAPI_ENV) >/dev/null 2>&1; export SYCL_CACHE_PERSISTENT=0;,)

# setvars.sh is a bash script; use bash so sourcing it works in recipes.
SHELL := /bin/bash

# Permite override: make bench MODEL=... NEW_TOKENS=... RUNS=...
MODEL      ?= Qwen/Qwen2.5-0.5B-Instruct
NEW_TOKENS ?= 128
RUNS       ?= 3
WARMUP     ?= 1
DTYPE      ?= bfloat16
DEVICE     ?= xpu
PROMPT_TOKENS ?=          # vazio = prompt natural; ex: 64,256,1024 para sweep de KV-cache
LLAMA_QUANT ?= q4_k_m
SERVE_HOST  ?= 127.0.0.1
SERVE_PORT  ?= 8080
SERVE_MODEL ?=           # vazio = perguntar interactivamente (só modelos que cabem na memória)
SERVE_NCTX  ?= 0          # 0 = automático (contexto de treino do modelo, limitado por SERVE_MAX_CTX)
SERVE_MAX_CTX ?= 32768    # tecto global do modo automático (modelos grandes têm limite próprio mais baixo na iGPU)

# Checkout/build do llama.cpp NATIVO (binário llama-server). Fora do repo (cache XDG)
# porque o build é pesado. Override com LLAMA_CPP_DIR=.
LLAMA_CPP_DIR ?= $(HOME)/.cache/llama/llama.cpp
LLAMA_CPP_REPO ?= https://github.com/ggml-org/llama.cpp
# cmake/ninja: usa os do .venv se existirem (instalados via pip), senão os do sistema.
CMAKE := $(if $(wildcard .venv/bin/cmake),$(abspath .venv/bin/cmake),cmake)
NATIVE_PATH := $(abspath .venv/bin):$(PATH)

.PHONY: help bench bench-quick sweep sweep-quick tokenizer check \
        test test-gpu test-hf test-chat all clean \
        setup-llama bench-llama bench-llama-quick sweep-llama sweep-llama-quick \
        setup-llama-server serve-llama opencode-config

help:
	@echo "Targets disponíveis:"
	@echo "  make bench         — benchmark único ($(MODEL), $(NEW_TOKENS) tokens, $(RUNS) runs)"
	@echo "  make bench-quick   — benchmark rápido (1 run, 64 tokens)"
	@echo "  make sweep         — sweep completo (modelos × dtypes × sizes)"
	@echo "  make sweep-quick   — sweep smoke test (1 combinação)"
	@echo "  make tokenizer     — benchmark de tokenizers (eficiência PT/EN/code)"
	@echo "  make check         — verifica acesso aos modelos HuggingFace"
	@echo "  make test          — corre todos os smoke tests (gpu + hf + chat)"
	@echo "  make test-gpu      — smoke test XPU"
	@echo "  make test-hf       — smoke test HuggingFace"
	@echo "  make test-chat     — smoke test chat"
	@echo "  make all           — check + bench + sweep + tokenizer"
	@echo "  make clean         — limpa caches e resultados"
	@echo ""
	@echo "── llama.cpp (motor GGUF quantizado) ──"
	@echo "  Não vem com 'uv sync': não há wheel GPU pré-built, o build SYCL"
	@echo "  é compilado a partir do código (oneAPI). Corre o setup UMA vez:"
	@echo "  make setup-llama        — bindings llama-cpp-python p/ benchmarks (SYCL, requer oneAPI)"
	@echo "  make bench-llama        — benchmark llama.cpp ($(MODEL), quant=$(LLAMA_QUANT))"
	@echo "  make bench-llama-quick  — benchmark llama.cpp rápido (1 run, 64 tokens)"
	@echo "  make sweep-llama        — sweep llama.cpp (modelos × quants × sizes)"
	@echo "  make sweep-llama-quick  — sweep llama.cpp smoke test"
	@echo ""
	@echo "── Servir um modelo ao opencode (llama-server --jinja: tool-calling nativo) ──"
	@echo "  make setup-llama-server — clona + compila o llama-server (SYCL, requer oneAPI)"
	@echo "  make serve-llama        — servidor OpenAI-compatível p/ opencode (escolhe modelo)"
	@echo "  make opencode-config    — imprime a config do provider do opencode (JSON)"
	@echo ""
	@echo "Variáveis: MODEL DTYPE NEW_TOKENS RUNS WARMUP DEVICE PROMPT_TOKENS LLAMA_QUANT"
	@echo "  PROMPT_TOKENS=64,256,1024  — adiciona sweep sobre tamanho do prompt/KV-cache"
	@echo "  SERVE_HOST=127.0.0.1 SERVE_PORT=8080  — endereço do serve-llama"
	@echo "  SERVE_MODEL=...  — modelo a servir (vazio = menu interactivo dos que cabem)"
	@echo "  SERVE_NCTX=0  — contexto do serve-llama (0=auto pelo modelo, limite SERVE_MAX_CTX)"
	@echo "  LLAMA_CPP_DIR=$(LLAMA_CPP_DIR)  — dir do checkout/build nativo do llama.cpp"

bench:
	$(PY) benchmarks/benchmark_tps.py \
		--model $(MODEL) --dtype $(DTYPE) \
		--new-tokens $(NEW_TOKENS) --runs $(RUNS) --warmup $(WARMUP) \
		--device $(DEVICE)$(if $(PROMPT_TOKENS), --prompt-tokens $(PROMPT_TOKENS),)

bench-quick:
	$(PY) benchmarks/benchmark_tps.py \
		--model $(MODEL) --new-tokens 64 --runs 1 --warmup 1

sweep:
	$(PY) benchmarks/bench_sweep.py$(if $(PROMPT_TOKENS), --prompt-tokens $(PROMPT_TOKENS),)

sweep-quick:
	$(PY) benchmarks/bench_sweep.py --quick

tokenizer:
	$(PY) benchmarks/benchmark_tokenizer.py

check:
	$(PY) utils/check_access.py

test: test-gpu test-hf test-chat

test-gpu:
	$(PY) tests/test_gpu.py

test-hf:
	$(PY) tests/test_hf.py

test-chat:
	$(PY) tests/test_chat.py

all: check bench sweep tokenizer

clean:
	@find . -type d -name __pycache__ -not -path './.venv/*' -exec rm -rf {} + 2>/dev/null || true
	@rm -f results/bench_results.md results/bench_results.csv results/tokenizer_results.md results/llama_results.md results/llama_results.csv
	@echo "✔  Limpeza concluída."

# ── llama.cpp (motor opcional) ──────────────────────────────────────────────
# Setup pontual: llama-cpp-python não está no pyproject (não há wheel GPU
# pré-built), por isso compila-se à parte com o backend SYCL para a Arc GPU
# (precisa do Intel oneAPI: DPC++ + MKL).

setup-llama:
	@if [ ! -f "$(ONEAPI_ENV)" ]; then \
		echo "✗ Intel oneAPI não encontrado em $(ONEAPI_ENV)."; \
		echo "  Instala o Intel oneAPI Base Toolkit (DPC++ + MKL) ou define"; \
		echo "  ONEAPI_ENV=/caminho/para/setvars.sh."; \
		exit 1; \
	fi
	@echo "A compilar llama-cpp-python com backend SYCL (Intel Arc GPU)..."
	$(SYCL_ENV) \
	CMAKE_ARGS="-DGGML_SYCL=on -DGGML_SYCL_TARGET=INTEL -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx" \
		uv pip install llama-cpp-python --force-reinstall --no-cache-dir
	@echo "✔  llama-cpp-python instalado (SYCL/GPU) — para 'make bench-llama'."

bench-llama:
	$(SYCL_ENV) $(PY) benchmarks/benchmark_llama.py \
		--model $(MODEL) --quant $(LLAMA_QUANT) \
		--new-tokens $(NEW_TOKENS) --runs $(RUNS) --warmup $(WARMUP)

bench-llama-quick:
	$(SYCL_ENV) $(PY) benchmarks/benchmark_llama.py \
		--model $(MODEL) --quant $(LLAMA_QUANT) \
		--new-tokens 64 --runs 1 --warmup 1

sweep-llama:
	$(SYCL_ENV) $(PY) benchmarks/bench_llama_sweep.py

sweep-llama-quick:
	$(SYCL_ENV) $(PY) benchmarks/bench_llama_sweep.py --quick

# Compila o binário llama-server (servidor C++ oficial) com o backend SYCL (Intel
# Arc). É o servidor usado para servir modelos ao opencode (tool-calling nativo).
setup-llama-server:
	@if [ ! -f "$(ONEAPI_ENV)" ]; then \
		echo "✗ Intel oneAPI não encontrado em $(ONEAPI_ENV)."; \
		echo "  Instala o Intel oneAPI Base Toolkit (DPC++ + MKL) ou define"; \
		echo "  ONEAPI_ENV=/caminho/para/setvars.sh."; \
		exit 1; \
	fi
	@if [ ! -d "$(LLAMA_CPP_DIR)/.git" ]; then \
		echo "A clonar llama.cpp para $(LLAMA_CPP_DIR)..."; \
		git clone --depth 1 $(LLAMA_CPP_REPO) "$(LLAMA_CPP_DIR)"; \
	else \
		echo "A actualizar $(LLAMA_CPP_DIR)..."; \
		git -C "$(LLAMA_CPP_DIR)" pull --ff-only || true; \
	fi
	@echo "A compilar llama-server (SYCL/Intel Arc GPU)..."
	$(SYCL_ENV) PATH="$(NATIVE_PATH)" $(CMAKE) -B "$(LLAMA_CPP_DIR)/build" -S "$(LLAMA_CPP_DIR)" \
		-G Ninja \
		-DGGML_SYCL=ON -DGGML_SYCL_TARGET=INTEL \
		-DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx \
		-DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF
	$(SYCL_ENV) PATH="$(NATIVE_PATH)" $(CMAKE) --build "$(LLAMA_CPP_DIR)/build" --config Release \
		-j --target llama-server
	@echo "A instalar dependências do launcher (gguf, psutil)..."
	uv pip install "gguf>=0.10.0" "psutil>=5.9.0"
	@echo "✔  llama-server compilado: $(LLAMA_CPP_DIR)/build/bin/llama-server"

# Servidor OpenAI-compatível na GPU para o opencode (ou qualquer cliente OpenAI),
# via llama-server --jinja (tool-calling nativo, template do GGUF).
# Endpoint: http://$(SERVE_HOST):$(SERVE_PORT)/v1  (apiKey ignorada).
serve-llama:
	$(SYCL_ENV) LLAMA_CPP_DIR="$(LLAMA_CPP_DIR)" $(PY) benchmarks/serve_llama_native.py \
		$(if $(SERVE_MODEL),--model $(SERVE_MODEL),) --quant $(LLAMA_QUANT) \
		--host $(SERVE_HOST) --port $(SERVE_PORT) \
		--n-ctx $(SERVE_NCTX) --max-ctx $(SERVE_MAX_CTX)

# Imprime a configuração do provider do opencode (JSON, gerada a partir do
# catálogo). Copia para ~/.config/opencode/opencode.json ou redirecciona:
#   make opencode-config > ~/.config/opencode/opencode.json
opencode-config:
	@$(PY) benchmarks/serve_llama_native.py --print-opencode-config \
		--host $(SERVE_HOST) --port $(SERVE_PORT)


