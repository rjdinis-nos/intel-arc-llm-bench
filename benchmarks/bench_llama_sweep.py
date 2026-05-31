"""
Sweep de benchmarks sobre benchmark_llama.py.

Varia (modelo × quant × new-tokens) e produz tabela markdown + CSV
comparável com bench_results.md (HF Transformers + XPU).

Uso:
  python bench_llama_sweep.py                 # sweep (SYCL/Intel Arc GPU)
  python bench_llama_sweep.py --quick         # smoke test (1 combinação)
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT  = SCRIPT_DIR.parent
LLAMA_SCRIPT = SCRIPT_DIR / "benchmark_llama.py"
RESULTS_DIR  = REPO_ROOT / "results"

# Modelo destacado no sweep (referência de desempenho de um modelo de código maior);
# incluído para se medir o desempenho mesmo não sendo o recomendado para serving.
RECOMMENDED_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"

DEFAULT_MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct",
    RECOMMENDED_MODEL,
]
DEFAULT_QUANTS = ["q4_k_m"]
DEFAULT_TOKENS = [128, 512]

RX = {
    "ttft":    re.compile(r"TTFT m[eé]dio\s*:\s*([\d.]+)\s*ms"),
    "prefill": re.compile(r"Prefill t/s\s*:\s*([\d.]+)"),
    "decode":  re.compile(r"Decode\s+t/s\s*:\s*([\d.]+)"),
    "overall": re.compile(r"Overall t/s\s*:\s*([\d.]+)"),
    "size_mb": re.compile(r"GGUF size\s*:\s*([\d.]+)\s*MB"),
}

LEGEND = """\
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

"""


@dataclass
class LlamaResult:
    model: str
    quant: str
    new_tokens: int
    ttft_ms: float | None
    prefill_tps: float | None
    decode_tps: float | None
    overall_tps: float | None
    gguf_mb: float | None
    error: str | None = None


def sanitize_note(s: str | None, max_len: int = 120) -> str:
    if not s:
        return ""
    s = re.sub(r"\s+", " ", s).strip().replace("|", "\\|")
    return s[: max_len - 1].rstrip() + "…" if len(s) > max_len else s


def run_one(
    model: str, quant: str, new_tokens: int,
    runs: int, warmup: int,
) -> LlamaResult:
    cmd = [
        sys.executable, str(LLAMA_SCRIPT),
        "--model", model,
        "--quant", quant,
        "--new-tokens", str(new_tokens),
        "--runs", str(runs),
        "--warmup", str(warmup),
    ]
    label = f"{model}  quant={quant}  new_tokens={new_tokens}"
    print(f"\n▶ {label}", flush=True)
    try:
        out = subprocess.run(
            cmd, capture_output=True, text=True, timeout=1200, check=False
        )
    except subprocess.TimeoutExpired:
        return LlamaResult(model, quant, new_tokens, None, None, None, None, None, "timeout")

    text = out.stdout + "\n" + out.stderr
    if out.returncode != 0:
        tail = "\n".join(text.strip().splitlines()[-4:])
        return LlamaResult(model, quant, new_tokens, None, None, None, None, None, tail)

    def grab(key: str) -> float | None:
        m = RX[key].search(text)
        return float(m.group(1)) if m else None

    r = LlamaResult(
        model=model, quant=quant, new_tokens=new_tokens,
        ttft_ms=grab("ttft"),
        prefill_tps=grab("prefill"),
        decode_tps=grab("decode"),
        overall_tps=grab("overall"),
        gguf_mb=grab("size_mb"),
    )
    print(
        f"  → TTFT={r.ttft_ms} ms  decode={r.decode_tps} t/s  "
        f"prefill={r.prefill_tps} t/s  gguf={r.gguf_mb} MB",
        flush=True,
    )
    return r


def to_markdown(results: list[LlamaResult]) -> str:
    header = "| Model | quant | new_tok | TTFT (ms) | Prefill t/s | **Decode t/s** | Overall t/s | GGUF MB | Notes |"
    sep    = "|---|---|---:|---:|---:|---:|---:|---:|---|"
    dash   = "—"
    lines  = [LEGEND, "## Resultados", "", header, sep]
    for r in results:
        notes   = sanitize_note(r.error)
        ttft    = dash if r.ttft_ms    is None else f"{r.ttft_ms:.1f}"
        prefill = dash if r.prefill_tps is None else f"{r.prefill_tps:.0f}"
        decode  = dash if r.decode_tps  is None else f"{r.decode_tps:.2f}"
        overall = dash if r.overall_tps is None else f"{r.overall_tps:.2f}"
        gguf    = dash if r.gguf_mb     is None else f"{r.gguf_mb:.0f}"
        lines.append(
            f"| `{r.model.split('/')[-1]}` | {r.quant.upper()} | {r.new_tokens} | "
            f"{ttft} | {prefill} | **{decode}** | {overall} | {gguf} | {notes} |"
        )
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--models",        help="csv de modelos")
    p.add_argument("--quants",        help="csv de quantizações (ex: q4_k_m,q8_0)")
    p.add_argument("--tokens",        help="csv de new-tokens (ex: 128,512)")
    p.add_argument("--runs",          type=int, default=3)
    p.add_argument("--warmup",        type=int, default=1)
    p.add_argument("--quick",         action="store_true",
                   help="smoke test: 1 modelo, 1 quant, 1 size")
    p.add_argument("--out",           default=str(RESULTS_DIR / "llama_results"),
                   help="prefixo dos ficheiros de saída")
    args = p.parse_args()

    if args.quick:
        models = [DEFAULT_MODELS[0]]
        quants = [DEFAULT_QUANTS[0]]
        tokens = [128]
    else:
        models = args.models.split(",") if args.models else DEFAULT_MODELS
        quants = args.quants.split(",") if args.quants else DEFAULT_QUANTS
        tokens = [int(x) for x in args.tokens.split(",")] if args.tokens else DEFAULT_TOKENS

    total = len(models) * len(quants) * len(tokens)
    print(f"== Sweep llama.cpp [SYCL/GPU]: {len(models)} modelos × {len(quants)} quants × {len(tokens)} sizes = {total} runs ==")

    results: list[LlamaResult] = []
    for m in models:
        for q in quants:
            for n in tokens:
                r = run_one(m, q, n, args.runs, args.warmup)
                results.append(r)

    md      = to_markdown(results)
    out_md  = Path(f"{args.out}.md")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(md + "\n", encoding="utf-8")

    with open(f"{args.out}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        w.writeheader()
        for r in results:
            w.writerow(asdict(r))

    print("\n" + md)
    print(f"\n✔  Resultados em {args.out}.md  e  {args.out}.csv")


if __name__ == "__main__":
    main()
