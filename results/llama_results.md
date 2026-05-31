# Benchmark Results — llama.cpp (SYCL / Intel Arc GPU)

**Legenda dos campos:**

| Campo | Descrição |
|---|---|
| **Model** | Modelo HuggingFace testado (nome curto). |
| **quant** | Quantização GGUF: `Q4_K_M` (4-bit, ~0.5 bytes/param), `Q8_0` (8-bit), `F16` (16-bit). |
| **new_tok** | Máximo de tokens gerados na fase de *decode*. |
| **TTFT (ms)** | *Time To First Token* — inclui prefill. |
| **Prefill t/s** | Débito da fase de prefill. |
| **Decode t/s** | Débito da fase de decode. **Métrica principal.** |
| **Overall t/s** | Débito total ponta-a-ponta. |
| **GGUF MB** | Tamanho do ficheiro GGUF em MB (indica VRAM/RAM necessária). |
| **Notes** | Erros ou avisos. |

> Backend: SYCL/GPU (Intel Arc). Comparar com `bench_results.md`
> (HF Transformers + XPU, fp16/bf16).


## Resultados

| Model | quant | new_tok | TTFT (ms) | Prefill t/s | **Decode t/s** | Overall t/s | GGUF MB | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `Qwen2.5-Coder-7B-Instruct` | Q4_K_M | 128 | 116.7 | 292 | **8.88** | 8.88 | 4466 |  |
| `Qwen2.5-Coder-7B-Instruct` | Q4_K_M | 512 | 121.5 | 281 | **8.26** | 8.26 | 4466 |  |
