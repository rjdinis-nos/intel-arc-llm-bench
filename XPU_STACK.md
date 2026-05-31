# Stack de Inferência LLM na Intel Arc XPU

Hardware de referência: **Intel Core Ultra 7 255H · Intel Arc iGPU (Xe2, 128 EU, ~16 GB LPDDR5 partilhada) · WSL2 · Arch Linux**

---

## 1. Arquitectura do stack

```
┌────────────────────────────────────────────────────────────┐
│  Python (transformers / AutoModelForCausalLM)              │
│  benchmark_tps.py — geração via TextIteratorStreamer        │
├────────────────────────────────────────────────────────────┤
│  PyTorch 2.12.0+xpu                                        │
│  • ops: torch.matmul, F.scaled_dot_product_attention, …    │
│  • device: tensors vivem em "xpu" device (Arc iGPU)        │
├────────────────────────────────────────────────────────────┤
│  Intel Extension for PyTorch (IPEX) — embutido no +xpu     │
│  • JIT fusion de ops XPU                                   │
│  • Suporte a bf16/fp16 nos motores XMX                     │
├────────────────────────────────────────────────────────────┤
│  Triton-XPU 3.7.1 — compilação de kernels custom           │
├────────────────────────────────────────────────────────────┤
│  Intel SYCL RT + Level Zero                                │
│  • Runtime SYCL: intel-sycl-rt 2025.3.2                   │
│  • OpenCL ICD: intel-opencl-rt 2025.3.2                   │
│  • PTI (Performance Counters): intel-pti 0.16.0            │
├────────────────────────────────────────────────────────────┤
│  GPU hardware: Intel Arc Xe2 (128 EU, XMX INT8/BF16)       │
└────────────────────────────────────────────────────────────┘
```

### Como o PyTorch chega ao hardware

O build `torch+xpu` inclui um **XPU backend** nativo que, ao receber tensores no device `"xpu"`, os delega ao Intel oneAPI via **Level Zero** (o equivalente XPU ao CUDA para NVIDIA). O Triton-XPU compila kernels de atenção e outras ops para SPIR-V e envia-os ao driver IGC (`intel-graphics-compiler`), que os traduz para ISA EU do Arc.

O caminho é:

```
Python tensor.to("xpu")
    → torch XPU backend (C++)
        → Level Zero / SYCL dispatch
            → Intel Graphics Compiler (IGC)
                → EU kernel na Arc iGPU
```

---

## 2. Opções de inferência disponíveis

### 2.1 HF Transformers + PyTorch XPU ✅ (funcional)

O caminho principal. Suporta modelos em `bf16` / `fp16` directamente na GPU.

```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="xpu",
)
```

- **Dtype recomendado:** `bfloat16` — os motores XMX do Arc Xe2 têm aceleração nativa para BF16; costuma ser mais rápido que fp16.
- **Limitação:** sem quantização embutida — os pesos ficam em fp16/bf16 (2 bytes/parâmetro).
- **Overhead de Python:** ~13–14 s de arranque para Python 3.14 na primeira invocação.

### 2.2 llama.cpp ✅ — GPU (SYCL)

Modelos GGUF quantizados (Q4_K_M ≈ 4.5 bits/param). Após a actualização dos
drivers Arc + Intel oneAPI, corre na GPU via build SYCL (todas as layers).

```bash
make setup-llama       # build SYCL → GPU Intel Arc (requer oneAPI)
make sweep-llama       # benchmark na GPU
```

### 2.3 llama.cpp com OpenCL ❌ (compila, não executa em WSL2)

Ver [SETUP.md](SETUP.md) secção 3.1. A GPU é visível via OpenCL mas os kernels GGML ficam suspensos.

### 2.4 IPEX-LLM (INT4 na XPU) ❌ (incompatibilidade de versões)

Ver [SETUP.md](SETUP.md) secção 3.3. Aguardar versão compatível com `transformers ≥5.x`.

---

### 2.5 HuggingFace Transformers vs llama.cpp — diferenças fundamentais

São duas abordagens diferentes para correr o mesmo modelo:

**HuggingFace Transformers + PyTorch XPU**
- Carrega o modelo em **pesos originais** (bf16/fp16) directamente para a GPU.
- Usa o motor de cálculo do PyTorch (SYCL/Level Zero no caso XPU).
- Requer mais VRAM: 2 bytes/parâmetro em bf16.
- Interface de alto nível: `AutoModelForCausalLM.from_pretrained(..., device_map="xpu")`.

**llama.cpp**
- Converte o modelo para **formato GGUF com quantização** (ex: Q4_K_M ≈ 0.56 bytes/param).
- Backend em C++ optimizado para CPU (AVX2, OpenMP) ou GPU (SYCL/OpenCL).
- Usa muito menos memória — o mesmo modelo de 1B params ocupa ~600 MB vs ~2 GB em bf16.
- Python acede via `llama-cpp-python` (binding ctypes sobre a lib C++).

**Comparação directa (Llama-3.2-1B, 128 tokens):**

| Métrica | HF Transformers (XPU bf16) | llama.cpp (CPU Q4_K_M) |
|---|---:|---:|
| Decode t/s | **16.7** | **6.1** |
| Memória modelo | ~2.5 GB GPU | ~1.3 GB RAM |
| Instalação | `uv sync` | `make setup-llama` |
| Compilação necessária | ❌ (pre-built wheels) | ❌ (pre-built wheel) |

**Quando usar cada um:**
- GPU disponível → HF + XPU (2–3× mais rápido no decode)
- Sem GPU / memória limitada → llama.cpp CPU (footprint muito menor)
- Modelos grandes (>4B) sem VRAM suficiente → llama.cpp com quantização Q4

---

## 3. Fases de inferência e bottlenecks

Uma inferência LLM tem duas fases com características opostas:

### 3.1 Prefill (processamento do prompt)

- O prompt inteiro é processado em **paralelo** — uma passagem forward com batch de `P` tokens.
- É **compute-bound**: o número de FLOPs cresce com `P²` (atenção) e `P × d_model` (FFN).
- Nos modelos testados: 676–1665 t/s dependendo do tamanho do modelo.
- O TTFT (Time To First Token) é dominado por esta fase.

### 3.2 Decode (geração autoregressiva)

- Gera **um token de cada vez**, reutilizando o KV-cache dos tokens anteriores.
- É **memory-bandwidth bound**: a cada passo lê todos os pesos do modelo (~`W` bytes) para produzir 1 token.
- O throughput máximo teórico de decode é:

$$\text{decode\_tps}_{max} \approx \frac{\text{BW}_{\text{GPU}}}{\text{W}_{\text{modelo}}}$$

Onde $\text{BW}_{\text{GPU}}$ é a largura de banda de memória e $\text{W}_{\text{modelo}}$ é o tamanho dos pesos em bytes.

Para o Arc iGPU (LPDDR5 ~70 GB/s partilhada CPU+GPU, na prática ~35–45 GB/s disponíveis para GPU):

| Modelo | Pesos bf16 | Decode t/s teórico | Medido |
|---|---:|---:|---:|
| Qwen2.5-0.5B | 942 MB | ~37–47 | **24** |
| Qwen2.5-1.5B | 2944 MB | ~12–15 | **12–14** |
| Llama-3.2-1B | 2357 MB | ~15–19 | **17–19** |

> Os valores medidos estão perto do limite teórico para o 1.5B e Llama-1B. O 0.5B fica abaixo — provavelmente overhead de framework relativo ao pequeno tamanho do modelo.

---

## 4. Dimensionamento de memória, storage e throughput

Ver **[SIZING.md](SIZING.md)** para tabelas detalhadas de:
- Memória GPU por modelo e dtype (pesos + KV-cache + activações)
- Storage em disco (safetensors bf16 e GGUF quantizado)
- Estimativas de decode t/s e cenários práticos
- Projecção de ganho com quantização INT4

---

## 5. Limitações específicas desta configuração

| Factor | Impacto |
|---|---|
| Memória partilhada CPU/GPU (LPDDR5) | GPU e CPU competem pela mesma BW (~70 GB/s total) |
| WSL2 overhead | Adiciona ~10–20% de overhead em operações de I/O e scheduling |
| Python 3.14 startup | ~13–14 s por invocação Python (overhead do interpretador) |
| Sem quantização bf16 → INT4 | Sem `ipex-llm`, pesos sempre a 2 bytes/param |
| Contexto máximo efectivo | KV-cache limita contextos longos; 2048 tokens é o default seguro |
| Arc iGPU sem ECC | Não recomendado para inferência de produção sem validação de resultados |

---

## 6. Como correr e interpretar os benchmarks

```bash
# Sweep completo (3 modelos × bf16/fp16 × 128/512 tokens)
make sweep

# Modelo específico
make bench MODEL=Qwen/Qwen2.5-1.5B-Instruct NEW_TOKENS=256 RUNS=5

# Comparação CPU vs XPU (llama.cpp CPU)
make sweep-llama

# Ver resultados
cat results/bench_results.md    # XPU
cat results/llama_results.md    # CPU
```

### Interpretar os resultados

- **Decode t/s** é a métrica principal para UX de chat. Abaixo de ~5 t/s a experiência é perceptivelmente lenta.
- **Prefill t/s** importa para prompts longos (RAG, sumários). O TTFT = `prompt_tokens / prefill_tps`.
- **Mem MB** no bench_results é estimado como `n_params × 2`; a memória real (medida via `torch.xpu.memory_allocated()`) será ligeiramente superior.
- A diferença entre `new_tokens=128` e `new_tokens=512` no mesmo modelo reflecte o overhead crescente do KV-cache na atenção.
