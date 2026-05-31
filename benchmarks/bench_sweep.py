"""
Sweep de benchmarks sobre benchmark_tps.py.

Varia (modelo × dtype × new-tokens × prompt-tokens) e produz tabela markdown + CSV.

Uso:
  python bench_sweep.py                                     # sweep por defeito
  python bench_sweep.py --quick                             # versão rápida (só smoke test)
  python bench_sweep.py --models A,B --dtypes bfloat16,float16 --tokens 128,512
  python bench_sweep.py --prompt-tokens 64,256,1024         # sweep sobre tamanho do KV-cache
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

# Resolve paths relative to the repo root (script lives in benchmarks/)
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
TPS_SCRIPT = SCRIPT_DIR / "benchmark_tps.py"
RESULTS_DIR = REPO_ROOT / "results"
# Make utils/ importable (for check_access)
sys.path.insert(0, str(REPO_ROOT / "utils"))


# Erros que não vale a pena repetir para outros dtypes/sizes do mesmo modelo
HARD_ERROR_PATTERNS = {
    "401 — gated model (HF access required)": re.compile(r"401 Client Error|gated repo|awaiting a review", re.I),
    "403 — forbidden":                        re.compile(r"403 Client Error", re.I),
    "404 — model not found":                  re.compile(r"404 Client Error|RepositoryNotFound", re.I),
    "OOM (device out of memory)":             re.compile(r"out of memory|OutOfMemoryError|XPU out of memory", re.I),
}


def classify_error(text: str) -> str | None:
    for label, rx in HARD_ERROR_PATTERNS.items():
        if rx.search(text):
            return label
    return None


def sanitize_note(s: str | None, max_len: int = 160) -> str:
    if not s:
        return ""
    # colapsa whitespace (remove \n que partem a tabela) e escapa pipes
    s = re.sub(r"\s+", " ", s).strip().replace("|", "\\|")
    if len(s) > max_len:
        s = s[: max_len - 1].rstrip() + "…"
    return s


DEFAULT_MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct",
]
DEFAULT_DTYPES = ["bfloat16", "float16"]
DEFAULT_TOKENS = [128, 512]

# Regex sobre as linhas finais do benchmark
RX = {
    "ttft": re.compile(r"TTFT médio\s*:\s*([\d.]+)\s*ms"),
    "prefill": re.compile(r"Prefill t/s\s*:\s*([\d.]+)"),
    "decode": re.compile(r"Decode\s+t/s\s*:\s*([\d.]+)"),
    "overall": re.compile(r"Overall t/s\s*:\s*([\d.]+)"),
    "mem": re.compile(r"Model mem \(\w+\)\s*:\s*([\d.]+)\s*MB"),
    "prompt": re.compile(r"Prompt tokens\s*:\s*(\d+)"),
}


@dataclass
class Result:
    model: str
    dtype: str
    new_tokens: int
    device: str
    ttft_ms: float | None
    prefill_tps: float | None
    decode_tps: float | None
    overall_tps: float | None
    model_mem_mb: float | None = None
    error: str | None = None
    prompt_tokens: int | None = None


def run_one(
    model: str,
    dtype: str,
    new_tokens: int,
    runs: int,
    warmup: int,
    device: str,
    prompt_tokens: int | None = None,
) -> Result:
    cmd = [
        sys.executable, str(TPS_SCRIPT),
        "--model", model,
        "--dtype", dtype,
        "--new-tokens", str(new_tokens),
        "--runs", str(runs),
        "--warmup", str(warmup),
        "--device", device,
    ]
    if prompt_tokens is not None:
        cmd += ["--prompt-tokens", str(prompt_tokens)]
    label_pt = f"  prompt_tokens={prompt_tokens}" if prompt_tokens is not None else ""
    print(f"\n▶ {model}  dtype={dtype}  new_tokens={new_tokens}{label_pt}  device={device}", flush=True)
    try:
        out = subprocess.run(
            cmd, capture_output=True, text=True, timeout=900, check=False
        )
    except subprocess.TimeoutExpired:
        return Result(model, dtype, new_tokens, device, None, None, None, None, "timeout")

    text = out.stdout + "\n" + out.stderr
    if out.returncode != 0:
        label = classify_error(text)
        if label is None:
            label = "\n".join(text.strip().splitlines()[-5:])
        return Result(model, dtype, new_tokens, device, None, None, None, None, label)

    def grab(key):
        m = RX[key].search(text)
        return float(m.group(1)) if m else None

    def grabi(key):
        m = RX[key].search(text)
        return int(m.group(1)) if m else None

    r = Result(
        model=model, dtype=dtype, new_tokens=new_tokens, device=device,
        ttft_ms=grab("ttft"),
        prefill_tps=grab("prefill"),
        decode_tps=grab("decode"),
        overall_tps=grab("overall"),
        model_mem_mb=grab("mem"),
        prompt_tokens=grabi("prompt"),
    )
    print(
        f"  → prompt_tok={r.prompt_tokens}  TTFT={r.ttft_ms} ms  prefill={r.prefill_tps}  decode={r.decode_tps}  overall={r.overall_tps}  mem={r.model_mem_mb} MB",
        flush=True,
    )
    return r


LEGEND = """\
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

"""


def to_markdown(results: list[Result]) -> str:
    header = "| Model | dtype | prompt_tok | new_tok | TTFT (ms) | Prefill t/s | **Decode t/s** | Overall t/s | Mem MB | Notes |"
    sep = "|---|---|---:|---:|---:|---:|---:|---:|---:|---|"
    dash = "—"
    lines = [LEGEND, "## Resultados", "", header, sep]
    for r in results:
        notes = sanitize_note(r.error)
        ptok = dash if r.prompt_tokens is None else str(r.prompt_tokens)
        ttft = dash if r.ttft_ms is None else f"{r.ttft_ms:.1f}"
        prefill = dash if r.prefill_tps is None else f"{r.prefill_tps:.0f}"
        decode = dash if r.decode_tps is None else f"{r.decode_tps:.2f}"
        overall = dash if r.overall_tps is None else f"{r.overall_tps:.2f}"
        mem = dash if r.model_mem_mb is None else f"{r.model_mem_mb:.0f}"
        lines.append(
            f"| `{r.model.split('/')[-1]}` | {r.dtype} | {ptok} | {r.new_tokens} | "
            f"{ttft} | {prefill} | **{decode}** | {overall} | {mem} | {notes} |"
        )
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", help="csv de modelos")
    p.add_argument("--dtypes", help="csv de dtypes (bfloat16,float16,float32)")
    p.add_argument("--tokens", help="csv de new-tokens (ex: 128,512,1024)")
    p.add_argument(
        "--prompt-tokens",
        default=None,
        help="csv de tamanhos de prompt sintético (ex: 64,256,1024); vazio = usar prompt natural",
    )
    p.add_argument("--device", default="xpu")
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--quick", action="store_true", help="smoke test (1 modelo, 1 dtype, 1 size)")
    p.add_argument("--out", default=str(RESULTS_DIR / "bench_results"),
                   help="prefixo dos ficheiros de saída")
    args = p.parse_args()

    if args.quick:
        models, dtypes, tokens = [DEFAULT_MODELS[0]], ["bfloat16"], [128]
        prompt_tokens_list: list[int | None] = [None]
    else:
        models = args.models.split(",") if args.models else DEFAULT_MODELS
        dtypes = args.dtypes.split(",") if args.dtypes else DEFAULT_DTYPES
        tokens = [int(x) for x in (args.tokens.split(",") if args.tokens else map(str, DEFAULT_TOKENS))]
        if args.prompt_tokens:
            prompt_tokens_list = [int(x) for x in args.prompt_tokens.split(",")]
        else:
            prompt_tokens_list = [None]  # usa prompt natural (sem flag)

    total = len(models) * len(dtypes) * len(tokens) * len(prompt_tokens_list)
    pt_label = f" × {len(prompt_tokens_list)} prompt_sizes" if prompt_tokens_list != [None] else ""
    print(f"== Sweep: {len(models)} modelos × {len(dtypes)} dtypes × {len(tokens)} sizes{pt_label} = {total} runs ==")

    # Preflight: verifica acesso aos modelos antes de carregar pesos.
    try:
        from check_access import check_model
        from huggingface_hub import HfApi
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
        import os as _os
        _api = HfApi()
        _token = _os.environ.get("HF_TOKEN") or _os.environ.get("HUGGING_FACE_HUB_TOKEN")
        access: dict[str, str | None] = {}
        print("\n== Preflight (HF access) ==")
        for m in models:
            ok, msg = check_model(_api, m, _token)
            print(f"  {'✔' if ok else '✗'}  {m}: {msg}")
            access[m] = None if ok else msg
    except Exception as e:
        print(f"  (preflight skipped: {e})")
        access = {m: None for m in models}

    results: list[Result] = []
    for m in models:
        skip_reason: str | None = access.get(m)
        for d in dtypes:
            for n in tokens:
                for pt in prompt_tokens_list:
                    if skip_reason is not None:
                        results.append(Result(m, d, n, args.device, None, None, None, None, skip_reason))
                        print(f"  ⤵  skipping {m} {d} {n} pt={pt}: {skip_reason}", flush=True)
                        continue
                    r = run_one(m, d, n, args.runs, args.warmup, args.device, prompt_tokens=pt)
                    results.append(r)
                    # se for erro "duro" (gated/not-found), não tentar mais combinações
                    if r.error and any(tag in r.error for tag in ("401", "403", "404")):
                        skip_reason = r.error

    md = to_markdown(results)
    out_md = Path(f"{args.out}.md")
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
