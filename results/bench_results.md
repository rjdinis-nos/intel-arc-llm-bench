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
| **Mem MB** | Memória do device ocupada pelos pesos do modelo após carregamento. Estimada a partir de `n_params × 2 bytes` (fp16/bf16). Igual para ambos os dtypes e independente de `new_tokens`. |

> Medições obtidas via `benchmark_tps.py` com *warmup* + média de N *runs* (default: 1 warmup + 3 runs). Valores reportados são médias.


## Resultados

| Model | dtype | new_tok | TTFT (ms) | Prefill t/s | **Decode t/s** | Overall t/s | Mem MB | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `Qwen2.5-0.5B-Instruct` | bfloat16 | 128 | 51.1 | 1665 | **24.09** | 24.05 | 942 |  |
| `Qwen2.5-0.5B-Instruct` | bfloat16 | 512 | 66.5 | 1323 | **18.37** | 18.36 | 942 |  |
| `Qwen2.5-0.5B-Instruct` | float16 | 128 | 82.9 | 1035 | **16.83** | 16.78 | 942 |  |
| `Qwen2.5-0.5B-Instruct` | float16 | 512 | 53.0 | 1620 | **22.41** | 22.40 | 942 |  |
| `Qwen2.5-1.5B-Instruct` | bfloat16 | 128 | 125.7 | 676 | **12.50** | 12.45 | 2944 |  |
| `Qwen2.5-1.5B-Instruct` | bfloat16 | 512 | 101.7 | 842 | **13.85** | 13.84 | 2944 |  |
| `Qwen2.5-1.5B-Instruct` | float16 | 128 | 111.5 | 765 | **13.76** | 13.70 | 2944 |  |
| `Qwen2.5-1.5B-Instruct` | float16 | 512 | 103.0 | 827 | **12.70** | 12.69 | 2944 |  |
| `Llama-3.2-1B-Instruct` | bfloat16 | 128 | 98.6 | 924 | **16.70** | 16.61 | 2357 |  |
| `Llama-3.2-1B-Instruct` | bfloat16 | 512 | 97.2 | 938 | **17.27** | 17.24 | 2357 |  |
| `Llama-3.2-1B-Instruct` | float16 | 128 | 88.5 | 1029 | **19.12** | 19.02 | 2357 |  |
| `Llama-3.2-1B-Instruct` | float16 | 512 | 92.7 | 984 | **16.04** | 16.03 | 2357 |  |
