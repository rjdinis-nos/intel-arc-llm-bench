# Setup & Findings — Intel Arc XPU LLM Bench

Ambiente: **Intel Core Ultra 7 255H · Intel Arc iGPU (0x7D51, 128 EU, 16 GB LPDDR5 partilhada) · WSL2 · Arch Linux**

---

## 1. O que funciona — HF Transformers + PyTorch XPU

O único caminho funcional para inferência em GPU é **PyTorch `torch+xpu` com HuggingFace Transformers**. O stack inteiro está pré-integrado no `pyproject.toml` e instala-se com um único comando. **Não é necessário compilar nada** — todos os pacotes instalam como *pre-built wheels*.

### Pré-requisitos do sistema (sistema novo)

Necessário apenas uma vez, antes de qualquer instalação Python:

```bash
# Drivers GPU Intel Arc: runtime de computação + Level Zero + ICD OpenCL
# Todos no repositório 'extra' do Arch — não requer AUR
sudo pacman -S intel-compute-runtime level-zero-loader ocl-icd
```

| Pacote | Função |
|---|---|
| `intel-compute-runtime` | Runtime GPU Intel (NEO) — necessário para `torch.xpu` |
| `level-zero-loader` | Level Zero ICD — caminho de baixo nível usado pelo PyTorch XPU |
| `ocl-icd` | OpenCL ICD loader — necessário para `clinfo` e verificação |

Verificar que a GPU é visível:

```bash
clinfo | grep "Platform Name"   # → Intel(R) OpenCL Graphics
```

### Instalar uv (se ainda não estiver instalado)

```bash
curl -Lf https://astral.sh/uv/install.sh | sh
# Ou via pacman:
sudo pacman -S uv
```

### Instalação do ambiente Python

```bash
# Criar venv e instalar todas as dependências (sem compilação)
# Faz download de pre-built wheels: torch+xpu, transformers, triton-xpu, accelerate, gradio, ...
uv sync

# Verificar XPU disponível
.venv/bin/python -c "import torch; print(torch.__version__, torch.xpu.is_available())"
# → 2.12.0+xpu  True

.venv/bin/python -c "import torch; print(torch.xpu.get_device_name(0))"
# → Intel(R) Graphics [0x7d51]
```

O `pyproject.toml` aponta `torch`, `torchvision`, `torchaudio` e `triton-xpu` para o índice `https://download.pytorch.org/whl/xpu`. O `uv sync` resolve e descarrega tudo automaticamente — tempo típico: 2–5 min dependendo da ligação.

### Dependências instaladas

| Pacote | Versão | Origem |
|---|---|---|
| `torch` | 2.12.0+xpu | pytorch.org/whl/xpu |
| `torchvision` | 0.27.0+xpu | pytorch.org/whl/xpu |
| `torchaudio` | 2.11.0+xpu | pytorch.org/whl/xpu |
| `triton-xpu` | 3.7.1 | pytorch.org/whl/xpu |
| `transformers` | 5.9.0 | PyPI |
| `accelerate` | 1.13.0 | PyPI |
| `huggingface-hub` | 1.16.1 | PyPI |
| `intel-sycl-rt` | 2025.3.2 | dependência transitiva (bundled no torch+xpu) |
| `intel-openmp` | 2025.3.2 | dependência transitiva (bundled no torch+xpu) |
| `intel-pti` | 0.16.0 | dependência transitiva (bundled no torch+xpu) |
| `onemkl-sycl-*` | 2025.3.x | dependência transitiva (bundled no torch+xpu) |

> Todas as dependências Intel são bundled no wheel do `torch+xpu` — não precisam de instalação separada.

### Correr benchmarks

```bash
make sweep           # sweep completo: 3 modelos × bf16/fp16 × 128/512 tokens
make bench           # modelo único (default: Qwen2.5-0.5B, bfloat16, 128 tokens)
make sweep-quick     # smoke test (1 combinação)
```

### Resultados obtidos (Intel Arc iGPU, WSL2)

| Model | dtype | new_tok | **Decode t/s** | Prefill t/s | TTFT (ms) |
|---|---|---:|---:|---:|---:|
| Qwen2.5-0.5B-Instruct | bfloat16 | 128 | **24.09** | 1665 | 51 |
| Qwen2.5-0.5B-Instruct | bfloat16 | 512 | **18.37** | 1323 | 67 |
| Qwen2.5-1.5B-Instruct | bfloat16 | 128 | **12.50** | 676 | 126 |
| Qwen2.5-1.5B-Instruct | bfloat16 | 512 | **13.85** | 842 | 102 |
| Llama-3.2-1B-Instruct | bfloat16 | 128 | **16.70** | 924 | 99 |
| Llama-3.2-1B-Instruct | float16 | 128 | **19.12** | 1029 | 89 |

> `bfloat16` é o dtype recomendado no Arc (motores XMX). Ver `results/bench_results.md` para tabela completa.

---

## 2. llama.cpp — GPU (SYCL)

Após a actualização dos drivers Intel Arc, o `llama-cpp-python` corre na GPU
via um **build SYCL** (`make setup-llama`, requer o Intel oneAPI DPC++/MKL).
Todas as layers são descarregadas para a GPU.

```bash
make setup-llama        # compila com SYCL → GPU Intel Arc (precisa de oneAPI)
make sweep-llama        # sweep na GPU: 3 modelos × Q4_K_M × 128/512 tokens
make sweep-llama-quick  # smoke test
```

> **Nota (Arc 140T / Xe-LPG + oneAPI 2026.0):** a *persistent device-code cache*
> do SYCL provoca um *segmentation fault* durante a inferência (em
> `ggml_sycl_op_get_rows`, ao ler kernels compilados do disco). O `setvars.sh`
> força `SYCL_CACHE_PERSISTENT=1`; por isso o `Makefile` exporta
> `SYCL_CACHE_PERSISTENT=0` depois de o carregar (e o `benchmark_llama.py` faz
> `os.environ.setdefault(...)` como salvaguarda). Sem esta opção a GPU bloqueia
> mal começa a gerar tokens.

### Servir o modelo ao opencode (ou outro cliente OpenAI)

`make serve-llama` arranca o servidor OpenAI-compatível do `llama-cpp-python`
na GPU. Expõe `/v1/chat/completions`, `/v1/models`, etc.

Sem argumentos, **pergunta interactivamente** qual o modelo a servir, mostrando
apenas os que cabem na memória disponível (estimativa pesos + KV-cache na iGPU
UMA):

```text
== serve-llama: escolher modelo ==
Memória disponível : 25.3 GiB  (orçamento seguro 22.8 GiB)
  #  Modelo                              Pesos    +KV      Total   Cabe
   1  Qwen/Qwen2.5-0.5B-Instruct            469 MB    384 MB   1365 MB  sim
   2  Qwen/Qwen2.5-1.5B-Instruct           1408 MB    896 MB   2816 MB  sim
   3  meta-llama/Llama-3.2-1B-Instruct     1277 MB   1024 MB   2813 MB  sim
Escolhe um modelo [1-3] (Enter = 1):
```

```bash
make serve-llama                                    # menu interactivo
make serve-llama SERVE_MODEL=Qwen/Qwen2.5-1.5B-Instruct # sem menu (modelo fixo)
make serve-llama SERVE_HOST=0.0.0.0 SERVE_PORT=9000  # outro endereço/porta
make serve-llama SERVE_NCTX=16384                    # contexto fixo (override)
```

Modelos que não cabem aparecem marcados com `✗` e não são seleccionáveis. Sem
terminal interactivo (pipe/CI) escolhe automaticamente o mais pequeno que cabe.

A janela de contexto é definida **automaticamente** a partir do contexto de
treino do modelo (lido dos metadados do GGUF, `*.context_length`), limitada a
`SERVE_MAX_CTX` (default 32768) e ao limite do próprio modelo na iGPU. Isto evita
o erro *"Requested tokens exceeded context window"* com clientes que enviam
prompts/ferramentas grandes (ex: opencode). A KV-cache e o buffer de cómputo do
grafo crescem com o contexto; na iGPU Arc (Xe-LPG) o backend SYCL tem um limite
de alocação única que o 7B/8B excedem com 32768 (*"Failed to create
llama_context"*) — por isso esses modelos têm um tecto próprio de 16384, enquanto
modelos mais pequenos (≤3B) correm com 32768 completos. Override: `SERVE_NCTX=<n>`
(contexto fixo) ou `SERVE_MAX_CTX=<n>` (tecto do auto).

Endpoint: `http://127.0.0.1:8080/v1` — a `apiKey` não é validada (usa qualquer
valor). Requer os extras de servidor (já incluídos): `uvicorn`, `fastapi`,
`pydantic-settings`, `sse-starlette`, `starlette-context`, `PyYAML`, `gguf`.

Para modelos **Qwen** (ChatML), o servidor activa automaticamente um handler de
*tool-calling* nativo do Qwen2.5 (`--chat-format auto` → `qwen2.5-tool-calling`).
Sem ele, o `llama-cpp-python` devolvia as *tool calls* como texto no `content` (o
opencode não as reconhecia) e rebentava com *"ASGI callable returned without
completing response"* no padrão do opencode (stream + `tool_choice:auto`). Os
handlers genéricos (ex. `chatml-function-calling`) ou descartam silenciosamente as
mensagens `role:tool` (o modelo nunca vê os resultados → ciclo infinito) ou usam um
formato em que o Qwen não foi treinado (texto a vazar). O handler nativo:
(1) renderiza o formato ChatML do Qwen2.5 (`<tools>` / `<tool_call>` /
`<tool_response>`) e parseia as *tool calls* para o array `tool_calls`, com
*fallback* para JSON em *markdown*; (2) suporta *streaming* de *tool calls* (gera
sem stream e reemite como chunks); (3) corrige pedidos com `content:null` em
mensagens `assistant` (que o opencode envia após uma *tool call*). Override:
`make serve-llama` aceita `--chat-format none` (usa o template do GGUF) ou um nome
explícito de handler. Para depurar, define `SERVE_DUMP_REQUESTS=/caminho/log` para
gravar pedidos e *outputs* do modelo.

Configura o **opencode** com um *custom provider* OpenAI-compatível. Cria/edita
`opencode.json` no projecto (ou `~/.config/opencode/opencode.json`). O `id` do
modelo tem de coincidir com o que está a ser servido (recomendado para opencode:
**Qwen2.5-Coder-3B-Instruct** — capaz em código e *tool-calling*, e corre com a
janela de contexto completa de 32768 na iGPU; o 7B é melhor mas fica limitado a
ctx 16384):

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "llama-arc": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "llama.cpp (Intel Arc GPU)",
      "options": {
        "baseURL": "http://127.0.0.1:8080/v1",
        "apiKey": "local"
      },
      "models": {
        "Qwen/Qwen2.5-Coder-3B-Instruct": {
          "name": "Qwen2.5-Coder-3B (Arc GPU)",
          "limit": { "context": 32768, "output": 4096 }
        }
      }
    }
  }
}
```

O `id` do modelo tem de coincidir com o modelo escolhido no `serve-llama`
(visível em `GET /v1/models`) — se servires outro, troca o `id` aqui. Depois
corre `/connect` → **Other** (id `llama-arc`) e `/models` no opencode para o
seleccionar.

#### Modelos no menu do `serve-llama`

| Modelo | Tipo | Pesos (Q4_K_M) |
|---|---|---:|
| `Qwen/Qwen2.5-Coder-7B-Instruct` | código (melhor, mas ctx≤16384 na iGPU) | 4466 MB |
| `Qwen/Qwen2.5-Coder-3B-Instruct` ★ | código, ctx 32768 (recomendado opencode) | 2007 MB |
| `Qwen/Qwen2.5-Coder-1.5B-Instruct` | código (leve) | 1066 MB |
| `Qwen/Qwen2.5-Coder-0.5B-Instruct` | código (mínimo) | 469 MB |
| `Qwen/Qwen2.5-3B-Instruct` | geral | 2007 MB |
| `Qwen/Qwen2.5-1.5B-Instruct` | geral | 1408 MB |
| `Qwen/Qwen2.5-0.5B-Instruct` | geral (mínimo) | 469 MB |
| `meta-llama/Llama-3.1-8B-Instruct` | geral (grande) | 4693 MB |
| `meta-llama/Llama-3.2-3B-Instruct` | geral | 1926 MB |
| `meta-llama/Llama-3.2-1B-Instruct` | geral (leve) | 1277 MB |

Modelos maiores são mais capazes mas mais lentos na iGPU. Para adicionar outros,
acrescenta uma entrada em `SERVE_CATALOG` (benchmarks/serve_llama.py) e em
`GGUF_REPOS` (benchmarks/benchmark_llama.py).

### Resultados llama.cpp CPU (Q4_K_M, Core Ultra 7 255H) — baseline histórico

| Model | Decode t/s | Prefill t/s | GGUF MB |
|---|---:|---:|---:|
| Qwen2.5-0.5B Q4_K_M | **19.0** | 927 | 469 |
| Qwen2.5-1.5B Q4_K_M | **5.2** | 273 | 1408 |
| Llama-3.2-1B Q4_K_M | **6.1** | 300 | 1277 |

### CPU vs XPU (decode t/s, 128 tokens)

| Model | CPU Q4_K_M | XPU bf16 | Ganho XPU |
|---|---:|---:|---:|
| Qwen2.5-0.5B | 19.0 | 24.1 | **1.3×** |
| Qwen2.5-1.5B | 5.2 | 12.5 | **2.4×** |
| Llama-3.2-1B | 6.1 | 16.7 | **2.7×** |

---

## 3. Findings — opções que não funcionam

### 3.1 llama.cpp com GPU (OpenCL)

**Tentativa:** compilar `llama-cpp-python` com `-DGGML_OPENCL=on` para usar a GPU Arc via OpenCL.

```bash
CMAKE_ARGS="-DGGML_OPENCL=on" uv pip install llama-cpp-python --reinstall --no-cache
```

**O que aconteceu:**

1. A compilação **correu com sucesso** — a biblioteca `libggml-opencl.so` foi produzida.
2. `clinfo` confirma **Intel OpenCL 3.0** disponível e a GPU detectada.
3. No entanto, ao carregar a biblioteca Python, o processo ficava **indefinidamente suspenso** na primeira inferência (`[run 1/1]` sem output).
4. Em WSL2, o runtime OpenCL da Intel (`intel-compute-runtime`) expõe a GPU ao Linux mas o acesso a kernels GPGPU de computação intensiva (GGML kernels) bloqueia silenciosamente — provavelmente por limitações da camada de virtualização WSL2 para compute shaders.

**Conclusão:** llama.cpp + OpenCL não é praticável em WSL2. O `torch+xpu` usa um caminho diferente (Level Zero / SYCL runtime pré-empacotado) que já está devidamente configurado para WSL2 pelos mantenedores Intel/PyTorch.

---

### 3.2 llama.cpp com GPU (SYCL/oneAPI) — RESOLVIDO

> **Atualização (drivers Arc + oneAPI):** após a actualização dos drivers Intel
> Arc e com o Intel oneAPI Base Toolkit (DPC++ + MKL) instalado, o build SYCL
> passa a compilar e a correr na GPU. Este é agora o único caminho do llama.cpp
> no projecto (`make setup-llama`). O registo abaixo é o estado anterior,
> quando os compiladores `icx`/`icpx` ainda não estavam disponíveis.

**Tentativa (estado anterior):** compilar com os compiladores Intel DPC++ para SYCL.

```bash
CMAKE_ARGS="-DGGML_SYCL=on -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx" \
  uv pip install llama-cpp-python --reinstall
```

**O que aconteceu:**

- Os compiladores `icx` / `icpx` (Intel oneAPI DPC++ Compiler) **não estão instalados** no sistema.
- `find / -name "icx"` não encontrou nada; `/opt/intel/oneapi/` não existe.
- O pacman só tem `intel-graphics-compiler-bin` (IGC — compilador de kernels GPU), não o SDK oneAPI completo.

**Para funcionar seria necessário:**
```bash
# Instalar Intel oneAPI Base Toolkit (>3 GB)
# https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html
```

**Conclusão (anterior):** sem o Intel oneAPI DPC++ Toolkit instalado não era
possível compilar com SYCL. Com o toolkit instalado após a actualização dos
drivers, o build SYCL passa a funcionar — ver secção 2 (`make setup-llama`).

---

### 3.3 IPEX-LLM para quantização INT4 na XPU

**Tentativa:** usar `ipex-llm` para inferência INT4 na GPU Arc, com ganho esperado em memória e throughput.

```bash
# Tentativa 1 — índice Intel (URL oficial)
uv pip install ipex-llm[xpu] \
  --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
# → 403 Forbidden. O índice Intel está inacessível.

# Tentativa 2 — PyPI directo
uv pip install ipex-llm
# → ipex-llm==2.2.0 instalado (9.3 MB)
```

**O que aconteceu:**

`ipex-llm 2.2.0` (a única versão no PyPI) foi desenhado para `transformers ~4.47`. O ambiente tem `transformers 5.9.0`. O import falha imediatamente com:

```
ImportError: cannot import name 'OffloadedCache' from 'transformers.cache_utils'
```

Adicionalmente, em `transformers 5.9.0` faltam também:
- `QuantizedCacheConfig` (removido/renomeado)
- `ExtensionsTrie` (removido)
- `isin_mps_friendly` (removido)

São 4 símbolos em falta — demasiados para um patch simples.

**Para funcionar seria necessário** fazer downgrade de `transformers` para `~4.47`, o que afectaria os benchmarks HF Transformers já realizados com `5.9.0`.

**Conclusão:** `ipex-llm` está bloqueado pela incompatibilidade `transformers 4.47 vs 5.9`. O índice Intel está em 403. Aguardar que a Intel lance uma versão compatível com `transformers 5.x`.
