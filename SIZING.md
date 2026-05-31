# Dimensionamento de Memória, Storage e Throughput

Hardware de referência: **Intel Core Ultra 7 255H · Intel Arc iGPU (Xe2, 128 EU, ~16 GB LPDDR5 partilhada) · WSL2 · Arch Linux**

Ver [XPU_STACK.md](XPU_STACK.md) para a arquitectura completa do stack.

---

## 1. Dimensionamento de memória GPU

### 1.0 O que é carregado em memória

Durante inferência há três categorias de memória alocadas no XPU:

**1. Pesos do modelo — estático, carregado uma vez**

São as matrizes de parâmetros: embeddings, Q/K/V/O projections, FFN, layer norms. Para Qwen2.5-0.5B em bf16:

$$500\text{M params} \times 2\,\text{B (bf16)} = 1\,\text{GB teórico} \approx 950\,\text{MB medido}$$

(O modelo tem ~475 M parâmetros reais — o "0.5B" é arredondado.) Fica residente enquanto o modelo está em memória.

**2. KV Cache — dinâmico, cresce com o contexto**

Para cada token no contexto, cada layer guarda os tensores K e V da atenção:

$$\text{KV} = n_\text{layers} \times 2 \times n_\text{kv\_heads} \times d_\text{head} \times N_\text{tokens} \times 2\,\text{B}$$

Qwen2.5-0.5B tem 24 layers, 2 KV heads (GQA), head_dim = 64:

$$24 \times 2 \times 2 \times 64 \times N \times 2\,\text{B} = 12\,288 \times N\,\text{bytes} \approx 12\,\text{KB/token}$$

| Contexto | KV cache |
|---:|---:|
| 128 tok | 1.5 MB |
| 1 024 tok | 12 MB |
| 8 192 tok | 96 MB |
| 32 768 tok | 384 MB |

Daí a memória reportada ser sempre 950 MB nos benchmarks — o KV cache para 32–1024 tokens é insignificante face aos pesos.

**3. Activações — temporárias, por forward pass**

Buffers intermédios que existem apenas durante o cálculo (attention scores, FFN intermediates). Para batch=1 e contexto pequeno, são ~10–50 MB e são libertados após cada decode step.

**O que `torch.xpu.memory_allocated()` mede:**

É chamado após `model.to(device)` mas antes da inferência — por isso captura apenas os pesos (categoria 1). O KV cache e activações acrescem durante a geração mas são demasiado pequenos (neste modelo/contexto) para aparecer na diferença. Em modelos maiores ou contextos longos, o KV cache pode competir com os pesos pelo mesmo pool de memória — daí técnicas como KV cache quantization e paged attention.

---

### 1.1 Pesos do modelo (peso dominante)

$$\text{Mem\_pesos (MB)} = \frac{N_{params} \times B_{dtype}}{1024^2}$$

| dtype | $B_{dtype}$ | bytes/param |
|---|---:|---:|
| float32 | 4 | 4.0 |
| bfloat16 / float16 | 2 | 2.0 |
| int8 | 1 | 1.0 |
| Q4_K_M (GGUF) | ~0.56 | ~0.56 |
| sym_int4 (IPEX) | ~0.5 | ~0.5 |

Exemplos:

| Modelo | Params | bf16 | int8 | Q4_K_M |
|---|---:|---:|---:|---:|
| Qwen2.5-0.5B | 494 M | **988 MB** | 494 MB | **277 MB** |
| Qwen2.5-1.5B | 1540 M | **3080 MB** | 1540 MB | **862 MB** |
| Llama-3.2-1B | 1235 M | **2470 MB** | 1235 MB | **692 MB** |
| Qwen2.5-7B | 7620 M | **15 240 MB** | 7620 MB | **4267 MB** |
| Llama-3.2-3B | 3210 M | **6420 MB** | 3210 MB | **1797 MB** |

> A memória medida nos benchmarks (942 MB para 0.5B, 2944 MB para 1.5B) inclui buffers de activação, embeddings e overhead de framework — ligeiramente acima do estimado acima.

### 1.2 KV-cache (memória de atenção durante geração)

Durante a geração, cada token no contexto ocupa memória para os key/value de cada cabeça de atenção:

$$\text{KV\_cache (MB)} = \frac{2 \times L \times H \times d_h \times C_{ctx} \times B_{dtype}}{1024^2}$$

Onde:
- $L$ = número de layers
- $H$ = número de cabeças KV (GQA: cabeças KV, não Q)
- $d_h$ = dimensão por cabeça = $d_{model} / H_{total}$
- $C_{ctx}$ = comprimento do contexto em tokens
- $B_{dtype}$ = 2 bytes para bf16/fp16

Para os modelos testados (contexto = 2048 tokens, bf16):

| Modelo | Layers | KV heads | $d_h$ | KV-cache 2K | KV-cache 8K |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B | 24 | 2 | 64 | **24 MB** | **96 MB** |
| Qwen2.5-1.5B | 28 | 2 | 128 | **57 MB** | **228 MB** |
| Llama-3.2-1B | 16 | 8 | 64 | **64 MB** | **256 MB** |
| Qwen2.5-7B | 28 | 4 | 128 | **114 MB** | **456 MB** |

> O KV-cache é pequeno face aos pesos — o limite real neste hardware é sempre a memória para pesos.

### 1.3 Memória total estimada

$$\text{Mem\_total} \approx \text{Mem\_pesos} + \text{KV\_cache} + \text{overhead\_fw} \, (\approx 200\text{–}500 \text{ MB})$$

Para o Arc iGPU com 16 GB partilhados (mas ~6–8 GB tipicamente disponíveis para GPU em WSL2):

| Modelo | bf16 total | Cabe? | Q4 total | Cabe? |
|---|---:|---:|---:|---:|
| Qwen2.5-0.5B | ~1.2 GB | ✅ | ~0.5 GB | ✅ |
| Qwen2.5-1.5B | ~3.3 GB | ✅ | ~1.1 GB | ✅ |
| Llama-3.2-1B | ~2.7 GB | ✅ | ~0.9 GB | ✅ |
| Qwen2.5-7B | ~15.5 GB | ⚠️ limite | ~4.5 GB | ✅ |
| Llama-3.2-3B | ~6.7 GB | ⚠️ marginal | ~2.0 GB | ✅ |

---

## 2. Dimensionamento de storage (modelos em disco)

### HuggingFace safetensors (bf16)

$$\text{Storage\_HF (GB)} \approx \frac{N_{params} \times 2}{1024^3}$$

| Modelo | Tamanho disco |
|---|---:|
| Qwen2.5-0.5B | ~1.0 GB |
| Qwen2.5-1.5B | ~3.1 GB |
| Llama-3.2-1B | ~2.5 GB |
| Qwen2.5-7B | ~15.2 GB |

### GGUF quantizado (llama.cpp)

| Quant | bits/param | Fórmula | 1B modelo |
|---|---:|---|---:|
| Q2_K | 2.6 | $N \times 0.325$ B | 325 MB |
| Q4_K_M | 4.5 | $N \times 0.562$ B | 562 MB |
| Q5_K_M | 5.5 | $N \times 0.688$ B | 688 MB |
| Q8_0 | 8.0 | $N \times 1.0$ B | 1000 MB |
| F16 | 16.0 | $N \times 2.0$ B | 2000 MB |

Os ficheiros GGUF locais estão em `.llama_cache/`:

```
.llama_cache/
  qwen2.5-0.5b-instruct-q4_k_m.gguf   469 MB
  qwen2.5-1.5b-instruct-q4_k_m.gguf  1408 MB
  Llama-3.2-1B-Instruct-Q4_K_M.gguf  1277 MB
```

---

## 3. Estimativa de throughput para diferentes cenários

### 3.1 Modelo de throughput de decode

O decode é limitado pela largura de banda de memória. A relação empírica observada:

$$\text{decode\_tps} \approx k \times \frac{\text{BW efect.}}{\text{Mem\_pesos}}$$

Com $k \approx 0.6$–$0.8$ (eficiência de utilização, overhead de framework).

Para o Arc iGPU (~35 GB/s efectivos para GPU em WSL2):

| Modelo (bf16) | Pesos | Decode t/s estimado | Medido |
|---|---:|---:|---:|
| Qwen2.5-0.5B | 988 MB | 21–28 t/s | **24 t/s** |
| Qwen2.5-1.5B | 3080 MB | 7–9 t/s | **12–14 t/s** ¹ |
| Qwen2.5-7B | 15 240 MB | 1.4–1.8 t/s | — |

> ¹ O 1.5B supera a estimativa — a GQA (Group Query Attention) com apenas 2 KV heads reduz drasticamente o tráfego de memória do KV-cache.

### 3.2 Tabela de cenários práticos (Arc iGPU bf16)

| Caso de uso | Modelo | new_tokens | Decode t/s | Tempo resposta | Adequado? |
|---|---|---:|---:|---:|---:|
| Chat rápido | Qwen2.5-0.5B | 128 | 24 | ~5 s | ✅ |
| Chat standard | Qwen2.5-1.5B | 256 | 13 | ~20 s | ✅ |
| Resposta longa | Qwen2.5-1.5B | 1024 | 13 | ~79 s | ⚠️ lento |
| Código curto | Llama-3.2-1B | 256 | 17 | ~15 s | ✅ |
| Sumário | Qwen2.5-7B | 512 | ~2 | ~256 s | ❌ |

### 3.3 Efeito do comprimento do contexto no prefill

O prefill cresce aproximadamente com $O(P \times d_{model})$ (dominado pela FFN) para prompts médios. Para prompts muito longos, a atenção ($O(P^2)$) começa a dominar:

| Prompt tokens | Qwen2.5-1.5B prefill t/s | TTFT estimado |
|---:|---:|---:|
| 128 | ~840 t/s | ~150 ms |
| 512 | ~750 t/s | ~680 ms |
| 2048 | ~500 t/s | ~4.1 s |
| 8192 | ~200 t/s | ~41 s |

> Os valores de 128 e 512 são medidos; os restantes são extrapolados.

### 3.4 Ganho de quantização (projecção para IPEX-LLM INT4)

Quando `ipex-llm` suportar `transformers 5.x`, a estimativa de ganho esperado:

| Métrica | bf16 | INT4 (sym_int4) | Ganho |
|---|---:|---:|---:|
| Pesos em memória (1.5B) | 3080 MB | ~864 MB | **3.6×** menos |
| Decode t/s teórico | 13 t/s | ~46 t/s | **~3.5×** |
| Modelos que cabem (8 GB) | até ~3.5B | até ~14B | **4× maior** |

> A quantização INT4 reduz os pesos a ~0.5 bytes/param, acelerando o decode proporcionalmente (mais pesos por ciclo de memória).
