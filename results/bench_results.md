# Benchmark Results — LLM Throughput Sweep

**Legenda dos campos:**

| Campo | Descrição |
|---|---|
| **Model** | Modelo HuggingFace testado (apenas o nome curto após `/`). |
| **dtype** | Precisão numérica dos pesos: `bfloat16` (recomendado em XPU/Xe2 — XMX engines), `float16` ou `float32`. |
| **new_tok** | Número máximo de tokens gerados na fase de *decode* (`--new-tokens`). |
| **TTFT (ms)** | *Time To First Token* — latência média entre o pedido e o 1.º token emitido. Inclui o *prefill* do prompt. |
| **Prefill t/s** | Débito da fase de *prefill* (processamento paralelo do prompt). Tipicamente compute-bound. |
| **Decode t/s** | Débito da fase de *decode* (geração autoregressiva, 1 token de cada vez). **Métrica principal** para UX de chat — tipicamente memory-bandwidth bound. |
| **Overall t/s** | Débito total = `(prompt_tokens + new_tokens) / total_time`. Reflete a experiência ponta-a-ponta. |
| **Notes** | Erros ou avisos (ex.: `401 gated`, `OOM`, `timeout`). Linhas com `—` indicam que essa medição não foi obtida. |

> Medições obtidas via `benchmark_tps.py` com *warmup* + média de N *runs* (default: 1 warmup + 3 runs). Valores reportados são médias.


## Resultados

| Model | dtype | new_tok | TTFT (ms) | Prefill t/s | **Decode t/s** | Overall t/s | Notes |
|---|---|---:|---:|---:|---:|---:|---|
| `Qwen2.5-0.5B-Instruct` | bfloat16 | 128 | 60.5 | 1405 | **24.80** | 24.71 |  |
