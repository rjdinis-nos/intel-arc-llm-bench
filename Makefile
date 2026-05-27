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

# Permite override: make bench MODEL=... NEW_TOKENS=... RUNS=...
MODEL      ?= Qwen/Qwen2.5-0.5B-Instruct
NEW_TOKENS ?= 128
RUNS       ?= 3
WARMUP     ?= 1
DTYPE      ?= bfloat16
DEVICE     ?= xpu

.PHONY: help bench bench-quick sweep sweep-quick tokenizer check \
        test test-gpu test-hf test-chat all clean

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
	@echo "Variáveis: MODEL DTYPE NEW_TOKENS RUNS WARMUP DEVICE"

bench:
	$(PY) benchmarks/benchmark_tps.py \
		--model $(MODEL) --dtype $(DTYPE) \
		--new-tokens $(NEW_TOKENS) --runs $(RUNS) --warmup $(WARMUP) \
		--device $(DEVICE)

bench-quick:
	$(PY) benchmarks/benchmark_tps.py \
		--model $(MODEL) --new-tokens 64 --runs 1 --warmup 1

sweep:
	$(PY) benchmarks/bench_sweep.py

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
	@rm -f results/bench_results.md results/bench_results.csv results/tokenizer_results.md
	@echo "✔  Limpeza concluída."
