# Benchmark Results — LLM Throughput Sweep

**Legenda dos campos:**

| Campo | Descrição |
|---|---|
| **Model** | Modelo HuggingFace testado (apenas o nome curto após `/`). |
| **dtype** | Precisão numérica dos pesos: `bfloat16` (recomendado em XPU/Xe2 — XMX engines), `float16` ou `float32`. |
| **prompt_tok** | Tokens de entrada processados na fase de *prefill*. Com `--prompt-tokens` usa prompt sintético de exactamente N tokens; sem esse flag, é o tamanho real do prompt com chat template. |
| **new_tok** | Número máximo de tokens gerados na fase de *decode* (`--new-tokens`). |
| **TTFT (ms)** | *Time To First Token* — latência média entre o pedido e o 1.º token emitido. Inclui o *prefill* do prompt. |
| **Prefill t/s** | Débito da fase de *prefill* (processamento paralelo do prompt). Tipicamente compute-bound. |
| **Decode t/s** | Débito da fase de *decode* (geração autoregressiva, 1 token de cada vez). **Métrica principal** para UX de chat — tipicamente memory-bandwidth bound. |
| **Overall t/s** | Débito total = `(prompt_tokens + new_tokens) / total_time`. Reflete a experiência ponta-a-ponta. |
| **Notes** | Erros ou avisos (ex.: `401 gated`, `OOM`, `timeout`). Linhas com `—` indicam que essa medição não foi obtida. |
| **Mem MB** | Memória do device ocupada pelos pesos do modelo após carregamento (`torch.xpu/cuda.memory_allocated`). Indica o mínimo de VRAM necessário para correr cada combinação. |

> Medições obtidas via `benchmark_tps.py` com *warmup* + média de N *runs* (default: 1 warmup + 3 runs). Valores reportados são médias.


## Resultados

| Model | dtype | prompt_tok | new_tok | TTFT (ms) | Prefill t/s | **Decode t/s** | Overall t/s | Mem MB | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `Qwen2.5-0.5B-Instruct` | bfloat16 | 32 | 128 | 48.5 | 660 | **25.52** | 25.48 | 950 |  |
| `Qwen2.5-0.5B-Instruct` | bfloat16 | 64 | 128 | 40.9 | 1563 | **29.18** | 29.14 | 950 |  |
| `Qwen2.5-0.5B-Instruct` | bfloat16 | 128 | 128 | 55.6 | 2328 | **26.39** | 26.30 | 950 |  |
| `Qwen2.5-0.5B-Instruct` | bfloat16 | 256 | 128 | 66.2 | 3868 | **27.28** | 27.11 | 950 |  |
| `Qwen2.5-0.5B-Instruct` | bfloat16 | 512 | 128 | 91.4 | 5608 | **24.81** | 24.57 | 950 |  |
| `Qwen2.5-0.5B-Instruct` | bfloat16 | 1024 | 128 | 163.0 | 6282 | **21.38** | 20.97 | 950 |  |
