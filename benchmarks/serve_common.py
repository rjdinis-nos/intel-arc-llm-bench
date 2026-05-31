"""Catálogo, dimensionamento e resolução de contexto — partilhado.

Usado por:
  - serve_llama_native.py (servidor nativo llama.cpp `llama-server --jinja`)
  - benchmark_llama.py     (resolução/obtenção dos ficheiros GGUF)

Deliberadamente SEM importar `llama_cpp`: o caminho nativo não depende dos bindings
Python, por isso toda a lógica de catálogo/memória/contexto vive aqui.
"""
from __future__ import annotations

import sys

# Limite de contexto em modo automático. O n_ctx influencia o tamanho da KV-cache
# (que na iGPU Arc/UMA sai da RAM do sistema), por isso há um tecto global. Alguns
# modelos têm ainda um limite mais baixo por causa do buffer de cómputo do grafo:
# no 7B/8B, com 32768 esse buffer (~3.6 GiB) excede o limite de alocação única do
# backend SYCL e a criação do contexto falha. Esses modelos trazem 'ctx_cap' no
# catálogo. Override global com --n-ctx ou --max-ctx.
AUTO_CTX_CAP = 32768
FALLBACK_CTX = 8192

# Margem fixa para buffers de compute/contexto do runtime, além de pesos + KV.
RUNTIME_OVERHEAD_MIB = 512

# Modelo recomendado para o servidor (llama-server --jinja) e para o opencode.
# O parser de tool-calling nativo do llama.cpp (peg-native) só extrai tags
# <tool_call> e NÃO tem fallback para JSON em blocos markdown ```json. O
# Qwen2.5-3B "geral" emite <tool_call> nativamente e funciona perfeitamente; os
# modelos Qwen2.5-Coder (modelos de código) tendem a embrulhar a chamada em
# ```json e o parser nativo não as recupera — por isso o modelo geral é o
# recomendado para tool-calling com o opencode.
RECOMMENDED_MODEL = "Qwen/Qwen2.5-3B-Instruct"

# Catálogo de modelos que o servidor sabe disponibilizar (subconjunto dos que o
# benchmark consegue obter em GGUF). Para cada um guardamos uma estimativa do
# tamanho dos pesos (GGUF Q4_K_M, MB) e os parâmetros de arquitectura usados para
# estimar a KV-cache: n.º de layers, n.º de KV-heads e dimensão de cada head.
# Serve para decidir, sem descarregar, quais modelos cabem na memória disponível.
# 'note' é uma etiqueta curta mostrada no menu. Valores obtidos dos config.json /
# tamanhos reais dos ficheiros GGUF Q4_K_M no HuggingFace.
SERVE_CATALOG: dict[str, dict] = {
    # Qwen2.5-Coder — código (tool-calling fenced em ```json: usa o modelo geral)
    "Qwen/Qwen2.5-Coder-7B-Instruct": dict(
        weights_mib=4466, n_layers=28, n_kv_heads=4, head_dim=128, ctx_cap=16384,
        note="código (melhor, mas ctx≤16384; tool-calls em ```json não extraídas)"),
    "Qwen/Qwen2.5-Coder-3B-Instruct": dict(
        weights_mib=2007, n_layers=36, n_kv_heads=2, head_dim=128,
        note="código, ctx 32768 (tool-calls em ```json não extraídas pelo parser nativo)"),
    "Qwen/Qwen2.5-Coder-1.5B-Instruct": dict(
        weights_mib=1066, n_layers=28, n_kv_heads=2, head_dim=128, note="código (leve)"),
    "Qwen/Qwen2.5-Coder-0.5B-Instruct": dict(
        weights_mib=469, n_layers=24, n_kv_heads=2, head_dim=64, note="código (mínimo)"),
    # Qwen2.5 — geral
    "Qwen/Qwen2.5-3B-Instruct": dict(
        weights_mib=2007, n_layers=36, n_kv_heads=2, head_dim=128,
        note="geral, ctx 32768 (recomendado p/ opencode)"),
    "Qwen/Qwen2.5-1.5B-Instruct": dict(
        weights_mib=1408, n_layers=28, n_kv_heads=2, head_dim=128, note="geral"),
    "Qwen/Qwen2.5-0.5B-Instruct": dict(
        weights_mib=469, n_layers=24, n_kv_heads=2, head_dim=64, note="geral (mínimo)"),
    # Llama 3.x — geral
    "meta-llama/Llama-3.1-8B-Instruct": dict(
        weights_mib=4693, n_layers=32, n_kv_heads=8, head_dim=128, ctx_cap=16384,
        note="geral (grande, ctx≤16384 na iGPU)"),
    "meta-llama/Llama-3.2-3B-Instruct": dict(
        weights_mib=1926, n_layers=28, n_kv_heads=8, head_dim=128, note="geral"),
    "meta-llama/Llama-3.2-1B-Instruct": dict(
        weights_mib=1277, n_layers=16, n_kv_heads=8, head_dim=64, note="geral (leve)"),
}


def get_n_ctx_train(gguf_path) -> int | None:
    """Lê o contexto de treino (`*.context_length`) dos metadados do GGUF.

    Devolve None se não for possível (gguf não instalado ou chave ausente).
    """
    try:
        from gguf import GGUFReader
    except ImportError:
        return None
    try:
        reader = GGUFReader(str(gguf_path))
        for key, field in reader.fields.items():
            if key.endswith("context_length"):
                return int(field.contents())
    except Exception:
        return None
    return None


def available_mib() -> float:
    """Memória disponível (MiB). Na iGPU Arc (UMA) a VRAM sai da RAM do sistema."""
    try:
        import psutil
        return psutil.virtual_memory().available / 1024 ** 2
    except Exception:
        try:
            with open("/proc/meminfo") as fh:
                for line in fh:
                    if line.startswith("MemAvailable:"):
                        return int(line.split()[1]) / 1024  # kB → MiB
        except Exception:
            pass
    return float("inf")  # sem informação: não filtra


def kv_cache_mib(spec: dict, n_ctx: int) -> float:
    """KV-cache estimada (MiB) para n_ctx tokens, em fp16 (2 bytes), K e V."""
    bytes_per_token = 2 * spec["n_layers"] * spec["n_kv_heads"] * spec["head_dim"] * 2
    return bytes_per_token * n_ctx / 1024 ** 2


def estimate_mib(spec: dict, n_ctx: int) -> float:
    """Memória total estimada (MiB): pesos + KV-cache + overhead do runtime."""
    return spec["weights_mib"] + kv_cache_mib(spec, n_ctx) + RUNTIME_OVERHEAD_MIB


def select_model_interactively(plan_ctx: int) -> str:
    """Mostra os modelos que cabem na memória e pede ao utilizador para escolher.

    plan_ctx é o contexto usado para dimensionar a KV-cache (pior caso = --max-ctx).
    """
    budget = available_mib()
    safe_budget = budget * 0.90  # margem de segurança

    rows = []
    for model_id, spec in SERVE_CATALOG.items():
        need = estimate_mib(spec, plan_ctx)
        rows.append((model_id, spec, need, need <= safe_budget))

    fitting = [r for r in rows if r[3]]

    print("== serve-llama: escolher modelo ==")
    print(f"Memória disponível : {budget/1024:.1f} GiB  "
          f"(orçamento seguro {safe_budget/1024:.1f} GiB)")
    print(f"Contexto p/ estimativa: {plan_ctx} tokens")
    print()
    print("  #  Modelo                              Pesos    +KV      Total   Cabe  Notas")
    print("  -  ----------------------------------  -------  -------  -------  ----  -----")
    index_map: dict[int, str] = {}
    recommended_idx: int | None = None
    n = 0
    for model_id, spec, need, fits in rows:
        star = " ★" if model_id == RECOMMENDED_MODEL else ""
        if fits:
            n += 1
            index_map[n] = model_id
            label = f"{n:>2}"
            if model_id == RECOMMENDED_MODEL:
                recommended_idx = n
        else:
            label = " ✗"
        note = spec.get("note", "")
        if model_id == RECOMMENDED_MODEL:
            note = f"{note} — recomendado p/ opencode".strip(" —")
        print(f"  {label}  {model_id + star:<34}  "
              f"{spec['weights_mib']:>5} MB  "
              f"{kv_cache_mib(spec, plan_ctx):>5.0f} MB  "
              f"{need:>5.0f} MB  {'sim' if fits else 'NÃO':<4}  {note}")
    print()

    if not fitting:
        raise SystemExit(
            "Nenhum modelo do catálogo cabe na memória disponível "
            f"({budget/1024:.1f} GiB). Liberta memória ou reduz SERVE_MAX_CTX."
        )

    # Default = modelo recomendado se couber, senão o maior que cabe (mais capaz).
    default = recommended_idx if recommended_idx is not None else \
        max(index_map, key=lambda i: estimate_mib(SERVE_CATALOG[index_map[i]], plan_ctx))

    if not sys.stdin.isatty():
        # Sem terminal interactivo (ex: pipe/CI): usa o default (recomendado se couber).
        choice = index_map[default]
        print(f"(stdin não interactivo — escolhido automaticamente: {choice})")
        return choice

    star = " ★" if index_map[default] == RECOMMENDED_MODEL else ""
    while True:
        raw = input(f"Escolhe um modelo [1-{n}] (Enter = {default}{star}): ").strip()
        if raw == "":
            return index_map[default]
        if raw.isdigit() and int(raw) in index_map:
            return index_map[int(raw)]
        print(f"  Opção inválida: '{raw}'. Indica um número entre 1 e {n}.")


def resolve_ctx(model_id: str, gguf_path, n_ctx: int, max_ctx: int) -> tuple[int, str]:
    """Resolve a janela de contexto efectiva e devolve (n_ctx, mensagem legível).

    n_ctx > 0  → usado tal e qual (manual).
    n_ctx == 0 → automático: contexto de treino do GGUF, limitado pelo mínimo entre
                 max_ctx (tecto global) e o ctx_cap do modelo (limite da iGPU).
    """
    if n_ctx > 0:
        return n_ctx, f"{n_ctx} (definido manualmente)"

    model_cap = SERVE_CATALOG.get(model_id, {}).get("ctx_cap")
    cap = max_ctx if model_cap is None else min(max_ctx, model_cap)
    trained = get_n_ctx_train(gguf_path)
    if trained:
        resolved = min(trained, cap)
        if resolved < trained:
            reason = "--max-ctx" if cap == max_ctx else "limite da iGPU p/ este modelo"
            capped = f" (limitado por {reason})"
        else:
            capped = ""
        return resolved, f"{resolved} (auto: treino={trained}{capped})"
    return cap, (f"{cap} (auto: contexto de treino indisponível, a usar tecto {cap})")


# ---------------------------------------------------------------------------
# Obtenção de ficheiros GGUF (partilhada com benchmark_llama.py). Mantida aqui,
# sem importar llama_cpp, para o servidor nativo poder resolver o GGUF sem
# depender dos bindings Python.
# ---------------------------------------------------------------------------
import os  # noqa: E402
from pathlib import Path  # noqa: E402

# GGUF cache. Segue o padrão XDG ($XDG_CACHE_HOME, default ~/.cache). Os ficheiros
# GGUF são grandes (vários GB) e a partição do repositório pode ser pequena.
# Override com LLAMA_CACHE_DIR.
_XDG_CACHE = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
CACHE_DIR = Path(os.environ.get("LLAMA_CACHE_DIR", _XDG_CACHE / "llama"))

# Mapeamento: modelo HF → (repo GGUF no HuggingFace, padrão no nome do ficheiro)
GGUF_REPOS: dict[str, tuple[str, str]] = {
    # Qwen2.5-Coder (código / tool-calling — recomendados para agentes como opencode)
    "Qwen/Qwen2.5-Coder-0.5B-Instruct": ("Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF", "q4_k_m"),
    "Qwen/Qwen2.5-Coder-1.5B-Instruct": ("Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF", "q4_k_m"),
    "Qwen/Qwen2.5-Coder-3B-Instruct":   ("Qwen/Qwen2.5-Coder-3B-Instruct-GGUF",   "q4_k_m"),
    # 7B: o repo oficial é multi-ficheiro (sharded); usa-se o single-file do bartowski.
    "Qwen/Qwen2.5-Coder-7B-Instruct":   ("bartowski/Qwen2.5-Coder-7B-Instruct-GGUF", "Q4_K_M"),
    # Qwen2.5 (geral)
    "Qwen/Qwen2.5-0.5B-Instruct":       ("Qwen/Qwen2.5-0.5B-Instruct-GGUF",  "q4_k_m"),
    "Qwen/Qwen2.5-1.5B-Instruct":       ("Qwen/Qwen2.5-1.5B-Instruct-GGUF",  "q4_k_m"),
    "Qwen/Qwen2.5-3B-Instruct":         ("Qwen/Qwen2.5-3B-Instruct-GGUF",    "q4_k_m"),
    # Llama 3.x (geral) — GGUF do bartowski (não gated)
    "meta-llama/Llama-3.2-1B-Instruct": ("bartowski/Llama-3.2-1B-Instruct-GGUF", "Q4_K_M"),
    "meta-llama/Llama-3.2-3B-Instruct": ("bartowski/Llama-3.2-3B-Instruct-GGUF", "Q4_K_M"),
    "meta-llama/Llama-3.1-8B-Instruct": ("bartowski/Meta-Llama-3.1-8B-Instruct-GGUF", "Q4_K_M"),
}


def _load_token() -> str | None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    try:
        from dotenv import load_dotenv
        load_dotenv()
        token = token or os.environ.get("HF_TOKEN")
    except ImportError:
        pass
    return token


def get_gguf_path(model_id: str, quant_pattern: str, token: str | None = None) -> Path:
    """Devolve o caminho local do ficheiro GGUF, descarregando-o se necessário."""
    from huggingface_hub import hf_hub_download, list_repo_files

    if model_id not in GGUF_REPOS:
        raise ValueError(
            f"Modelo '{model_id}' não suportado. Adiciona uma entrada em GGUF_REPOS."
        )
    repo_id, default_pattern = GGUF_REPOS[model_id]
    pattern = quant_pattern or default_pattern

    # Caminho rápido: verificar cache local antes de ir à rede.
    # hf_hub_download faz sempre um HEAD request para validar o etag,
    # o que pode bloquear vários minutos se a ligação for instável.
    cache_slug = repo_id.replace("/", "--")
    if CACHE_DIR.exists():
        cached = [
            f for f in CACHE_DIR.rglob("*.gguf")
            if pattern.lower() in f.name.lower() and cache_slug in str(f)
        ]
        if cached:
            cached.sort(key=lambda f: ("large" in f.name.lower(), f.name))
            return cached[0]

    # Ficheiro não está em cache — descarregar do HF Hub.
    files = [f for f in list_repo_files(repo_id, token=token) if f.endswith(".gguf")]
    matches = [f for f in files if pattern.lower() in f.lower()]
    if not matches:
        raise FileNotFoundError(
            f"Nenhum ficheiro GGUF com '{pattern}' em {repo_id}.\n"
            f"Disponíveis: {files}"
        )
    # Se houver vários, prefere o mais pequeno (sem 'large' no nome)
    matches.sort(key=lambda f: ("large" in f.lower(), f))
    chosen = matches[0]

    local = hf_hub_download(
        repo_id=repo_id, filename=chosen,
        cache_dir=str(CACHE_DIR), token=token,
    )
    return Path(local)
