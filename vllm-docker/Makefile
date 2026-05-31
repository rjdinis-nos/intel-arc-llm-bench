# vLLM Docker (Intel XPU) — Makefile
# ----------------------------------
# Usage:
#   make             # show help
#   make build       # build the vLLM XPU image from source (one-time, slow)
#   make gpu-check   # verify the Intel XPU is visible inside the image
#   make start       # start the vLLM server (auto-detects WSL2 vs native /dev/dri)
#   make stop        # stop the vLLM server
#   make logs        # follow container logs
#   make status      # check if server is running
#   make health      # health check + list models
#   make test        # test the API with a sample request
#   make config      # show resolved compose configuration
#   make cleanup     # stop and remove volumes (deletes cached models)

# Load .env if present so overrides work for both compose and this Makefile.
ifneq (,$(wildcard ./.env))
include .env
export
endif

MODEL_NAME ?= Qwen/Qwen2.5-0.5B-Instruct
VLLM_PORT  ?= 8000
VLLM_REF   ?= v0.22.0
VLLM_IMAGE ?= vllm-xpu:$(VLLM_REF)
VLLM_SRC   ?= .vllm-src

# Detect GPU passthrough: WSL2 (/dev/dxg) vs native Linux (/dev/dri).
IS_WSL := $(shell grep -qi microsoft /proc/version 2>/dev/null && echo 1)
ifeq ($(IS_WSL),1)
  GPU_OVERRIDE := docker-compose.wsl.yml
else
  GPU_OVERRIDE := docker-compose.dri.yml
endif
COMPOSE := docker compose -f docker-compose.yml -f $(GPU_OVERRIDE)

.PHONY: help build gpu-check start stop restart logs status health test config cleanup

help:
	@echo "vLLM Docker (Intel XPU) — targets:"
	@echo "  make build      — build the vLLM XPU image from source ($(VLLM_REF), slow)"
	@echo "  make gpu-check  — verify the Intel XPU is visible inside the image"
	@echo "  make start      — start the vLLM server (GPU override: $(GPU_OVERRIDE))"
	@echo "  make stop       — stop the vLLM server"
	@echo "  make restart    — restart the vLLM server"
	@echo "  make logs       — follow container logs"
	@echo "  make status     — check if server is running"
	@echo "  make health     — health check + list models"
	@echo "  make test       — test the API with a sample request"
	@echo "  make config     — show resolved compose configuration"
	@echo "  make cleanup    — stop and remove volumes (deletes cached models)"
	@echo ""
	@echo "Variables: MODEL_NAME VLLM_PORT VLLM_REF VLLM_IMAGE"
	@echo "  e.g. make build VLLM_REF=v0.22.0"
	@echo "       make start MODEL_NAME=Qwen/Qwen2.5-1.5B-Instruct"

build:
	@echo "Building $(VLLM_IMAGE) from vLLM $(VLLM_REF) (this takes a while)..."
	@if [ ! -d "$(VLLM_SRC)/.git" ]; then \
		git clone --depth 1 --branch $(VLLM_REF) \
			https://github.com/vllm-project/vllm.git $(VLLM_SRC); \
	else \
		git -C $(VLLM_SRC) fetch --depth 1 origin tag $(VLLM_REF) && \
		git -C $(VLLM_SRC) checkout $(VLLM_REF); \
	fi
	docker build \
		-f $(VLLM_SRC)/docker/Dockerfile.xpu \
		--target vllm-openai \
		--shm-size=4g \
		-t $(VLLM_IMAGE) \
		$(VLLM_SRC)
	@echo "✓ Built $(VLLM_IMAGE)"

gpu-check:
	@VLLM_IMAGE=$(VLLM_IMAGE) ./gpu-check.sh

start:
	@echo "Starting vLLM server (override: $(GPU_OVERRIDE))..."
	@if [ "$(IS_WSL)" = "1" ]; then \
		echo "⚠️  WSL2 detected — Intel GPU compute in WSL2 is unreliable (see ../SETUP.md)."; \
	fi
	$(COMPOSE) up -d
	@echo "✓ Server starting. Check logs with: make logs"

stop:
	@echo "Stopping vLLM server..."
	$(COMPOSE) down
	@echo "✓ Server stopped"

restart: stop start

logs:
	$(COMPOSE) logs -f vllm

status:
	@if docker ps | grep -q vllm-server; then \
		echo "✓ vLLM server is running"; \
		docker ps --filter "name=vllm-server" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"; \
	else \
		echo "✗ vLLM server is not running"; \
		exit 1; \
	fi

health:
	@echo "Checking vLLM health..."
	@if curl -s http://localhost:$(VLLM_PORT)/health > /dev/null 2>&1; then \
		echo "✓ Server is healthy"; \
		curl -s http://localhost:$(VLLM_PORT)/v1/models | python -m json.tool; \
	else \
		echo "✗ Server is not responding (still loading? try: make logs)"; \
		exit 1; \
	fi

test:
	@echo "Testing vLLM API..."
	@curl -s http://localhost:$(VLLM_PORT)/v1/chat/completions \
		-H "Content-Type: application/json" \
		-d '{"model": "$(MODEL_NAME)", "messages": [{"role": "user", "content": "Say '\''Hello from vLLM'\'' in one line."}], "max_tokens": 50, "temperature": 0.7}' \
		| python -m json.tool

config:
	$(COMPOSE) config

cleanup:
	@read -p "Remove volumes and delete cached models? (y/N) " confirm; \
	if [ "$$confirm" = "y" ] || [ "$$confirm" = "Y" ]; then \
		echo "Cleaning up..."; \
		$(COMPOSE) down -v; \
		echo "✓ Cleanup complete"; \
	else \
		echo "Cancelled"; \
	fi
