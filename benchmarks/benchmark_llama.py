"""
Benchmark de débito com llama-cpp-python no backend SYCL (Intel Arc GPU).

Complementa benchmark_tps.py (HF Transformers + XPU) para comparação directa.
Os modelos são descarregados em formato GGUF do HuggingFace Hub.

Descarrega todas as layers para a GPU Intel Arc via SYCL. Requer um build SYCL
do llama-cpp-python (ver `make setup-llama`) e o runtime oneAPI no ambiente.

Uso:
  python benchmark_llama.py
  python benchmark_llama.py --model Qwen/Qwen2.5-1.5B-Instruct --quant q4_k_m
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
import time
from pathlib import Path

# Intel Arc 140T (Xe-LPG) + oneAPI 2026.0: the SYCL persistent device-code
# (JIT) cache segfaults while reading cached kernels from disk during
# ggml_sycl_op_get_rows. Disabling it avoids the crash. Set before the SYCL
# runtime initializes (i.e. before importing/using llama_cpp).
os.environ.setdefault("SYCL_CACHE_PERSISTENT", "0")

try:
    from llama_cpp import Llama
except ImportError:
    raise SystemExit(
        "llama-cpp-python não está instalado.\n"
        "  SYCL/GPU (Intel Arc): make setup-llama   "
        "(requer Intel oneAPI: source /opt/intel/oneapi/setvars.sh)\n"
        "                       CMAKE_ARGS=\"-DGGML_SYCL=on -DGGML_SYCL_TARGET=INTEL "
        "-DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx\" "
        "pip install llama-cpp-python --force-reinstall"
    )

from huggingface_hub import hf_hub_download, list_repo_files

REPO_ROOT = Path(__file__).resolve().parent.parent

# GGUF cache. Segue o padrão XDG ($XDG_CACHE_HOME, default ~/.cache) em vez de
# viver dentro do repositório: os ficheiros GGUF são grandes (vários GB) e a
# partição do repositório pode ser pequena. Override com LLAMA_CACHE_DIR.
_XDG_CACHE = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
CACHE_DIR  = Path(os.environ.get("LLAMA_CACHE_DIR", _XDG_CACHE / "llama"))

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

DEFAULT_PROMPT = (
    "Explain clearly and in detail what is large language model inference, "
    "including the prefill and decode phases, and why throughput (tokens per second) "
    "is an important metric."
)


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


def run_once(
    llm: Llama, prompt: str, new_tokens: int
) -> tuple[int, int, float, float]:
    """
    Retorna (prompt_tokens, generated_tokens, ttft_s, total_s).
    TTFT = tempo até ao 1.º token gerado (inclui prefill).
    """
    prompt_tokens = len(llm.tokenize(prompt.encode(), add_bos=True))

    t0 = time.perf_counter()
    ttft: float | None = None
    n_gen = 0

    for chunk in llm(prompt, max_tokens=new_tokens, stream=True, echo=False):
        if ttft is None:
            ttft = time.perf_counter() - t0
        n_gen += 1

    total = time.perf_counter() - t0
    if ttft is None:
        ttft = total

    return prompt_tokens, n_gen, ttft, total


def fmt_run(pt: int, ng: int, ttft: float, total: float) -> str:
    decode_s  = max(total - ttft, 1e-9)
    prefill   = pt / max(ttft, 1e-9)
    decode    = max(ng - 1, 0) / decode_s
    overall   = ng / max(total, 1e-9)
    return (
        f"prompt={pt:>4}t  gen={ng:>4}t  "
        f"TTFT={ttft*1000:7.1f}ms  total={total:6.2f}s  "
        f"prefill={prefill:7.2f} t/s  decode={decode:7.2f} t/s  "
        f"overall={overall:7.2f} t/s"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark llama.cpp (SYCL/Intel Arc GPU)")
    parser.add_argument("--model",        default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--quant",        default="q4_k_m",
                        help="Padrão de quantização (ex: q4_k_m, q8_0, f16)")
    parser.add_argument("--new-tokens",   type=int, default=128)
    parser.add_argument("--runs",         type=int, default=3)
    parser.add_argument("--warmup",       type=int, default=1)
    parser.add_argument("--n-threads",    type=int, default=None,
                        help="Threads CPU para amostragem/prompt (default: todos os cores)")
    parser.add_argument("--n-ctx",        type=int, default=2048)
    args = parser.parse_args()

    n_threads = args.n_threads or multiprocessing.cpu_count()
    backend   = "SYCL/GPU (Intel Arc)"
    token     = _load_token()

    print("== Benchmark llama.cpp ==")
    print(f"Modelo  : {args.model}")
    print(f"Quant   : {args.quant.upper()}")
    print(f"Backend : {backend}")
    print(f"Runs    : {args.runs} (warmup={args.warmup})")
    print(f"Tokens  : max_new_tokens={args.new_tokens}")
    print()

    print("A obter ficheiro GGUF...")
    gguf_path = get_gguf_path(args.model, args.quant, token)
    size_mb   = gguf_path.stat().st_size / 1024**2
    print(f"Ficheiro: {gguf_path.name} ({size_mb:.0f} MB)")
    print()

    print("A carregar modelo...")
    t0  = time.perf_counter()
    llm = Llama(
        model_path=str(gguf_path),
        n_gpu_layers=-1,
        n_threads=n_threads,
        n_ctx=args.n_ctx,
        verbose=False,
    )
    print(f"Carregado em {time.perf_counter() - t0:.2f}s\n")

    prompt = DEFAULT_PROMPT

    for i in range(args.warmup):
        print(f"[warmup {i + 1}/{args.warmup}] ", end="", flush=True)
        print(fmt_run(*run_once(llm, prompt, args.new_tokens)))

    results = []
    for i in range(args.runs):
        print(f"[run    {i + 1}/{args.runs}] ", end="", flush=True)
        r = run_once(llm, prompt, args.new_tokens)
        results.append(r)
        print(fmt_run(*r))

    if not results:
        return

    n = len(results)
    avg_ttft = sum(r[2] for r in results) / n
    avg_pre  = sum(r[0] / max(r[2], 1e-9) for r in results) / n
    avg_dec  = sum(max(r[1] - 1, 0) / max(r[3] - r[2], 1e-9) for r in results) / n
    avg_ovr  = sum(r[1] / max(r[3], 1e-9) for r in results) / n

    print()
    print(f"== Médias ({n} runs) ==")
    print(f"TTFT médio   : {avg_ttft * 1000:.1f} ms")
    print(f"Prefill t/s  : {avg_pre:.2f}")
    print(f"Decode  t/s  : {avg_dec:.2f}")
    print(f"Overall t/s  : {avg_ovr:.2f}")
    print(f"Quant        : {args.quant.upper()}")
    print(f"GGUF size    : {size_mb:.0f} MB")


if __name__ == "__main__":
    main()
