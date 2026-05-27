"""
Benchmark de tokenizers HuggingFace.

Mede para vários modelos:
  - Velocidade de encode (single + batch)
  - Velocidade de decode
  - Eficiência (chars/token, bytes/token) por língua: PT, EN, código

Uso:
  python benchmark_tokenizer.py
  python benchmark_tokenizer.py --models Qwen/Qwen2.5-0.5B-Instruct,gpt2
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

from transformers import AutoTokenizer

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


DEFAULT_MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "microsoft/Phi-3.5-mini-instruct",
    "gpt2",
]

SAMPLES = {
    "pt": (
        "A inferência de um modelo de linguagem é um processo em duas fases: "
        "prefill, onde o prompt completo é processado em paralelo e produz o "
        "primeiro token; e decode, onde cada token novo é gerado sequencialmente "
        "a partir do contexto acumulado. A eficiência depende da arquitetura, "
        "do hardware acelerador e da precisão numérica utilizada. Acentuação "
        "portuguesa: ção, são, não, coração, atenção, comunicação."
    ),
    "en": (
        "Large language model inference happens in two phases: prefill, where "
        "the entire prompt is processed in parallel to produce the first token; "
        "and decode, where each new token is generated sequentially from the "
        "accumulated context. Efficiency depends on the architecture, the "
        "accelerator hardware, and the numerical precision used."
    ),
    "code": (
        "def fibonacci(n: int) -> int:\n"
        "    a, b = 0, 1\n"
        "    for _ in range(n):\n"
        "        a, b = b, a + b\n"
        "    return a\n\n"
        "if __name__ == '__main__':\n"
        "    print([fibonacci(i) for i in range(10)])\n"
    ),
}


@dataclass
class Row:
    model: str
    is_fast: bool
    vocab: int
    # per-language efficiency (chars/token)
    cpt_pt: float
    cpt_en: float
    cpt_code: float
    # throughput
    encode_chars_per_s: float    # single-string encode
    batch_chars_per_s: float     # batch of 32
    decode_tokens_per_s: float   # detokenize


def time_loop(fn, iters: int) -> float:
    """Return seconds per iter (median-ish: mean of 3 inner runs)."""
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        times.append((time.perf_counter() - t0) / iters)
    return min(times)


def bench(model: str) -> Row | None:
    print(f"\n▶ {model}")
    try:
        tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    except Exception as e:
        print(f"   ⚠ falhou: {e!s:.120}")
        return None

    # efficiency
    eff = {}
    for lang, text in SAMPLES.items():
        ids = tok.encode(text, add_special_tokens=False)
        eff[lang] = len(text) / max(len(ids), 1)
    print(f"   vocab={tok.vocab_size}  fast={tok.is_fast}  "
          f"chars/token  pt={eff['pt']:.2f}  en={eff['en']:.2f}  code={eff['code']:.2f}")

    # encode single (warmup + measure)
    text = SAMPLES["pt"]
    tok.encode(text, add_special_tokens=False)
    sec = time_loop(lambda: tok.encode(text, add_special_tokens=False), iters=200)
    encode_cps = len(text) / sec

    # encode batch of 32
    batch = [text] * 32
    sec_b = time_loop(lambda: tok(batch, add_special_tokens=False), iters=50)
    batch_cps = (len(text) * 32) / sec_b

    # decode
    ids = tok.encode(text, add_special_tokens=False)
    sec_d = time_loop(lambda: tok.decode(ids, skip_special_tokens=True), iters=200)
    decode_tps = len(ids) / sec_d

    print(f"   encode={encode_cps/1e6:.2f} M chars/s   "
          f"batch32={batch_cps/1e6:.2f} M chars/s   "
          f"decode={decode_tps/1e6:.2f} M tokens/s")

    return Row(
        model=model, is_fast=tok.is_fast, vocab=tok.vocab_size,
        cpt_pt=eff["pt"], cpt_en=eff["en"], cpt_code=eff["code"],
        encode_chars_per_s=encode_cps,
        batch_chars_per_s=batch_cps,
        decode_tokens_per_s=decode_tps,
    )


def to_markdown(rows: list[Row]) -> str:
    h = ("| Model | Vocab | Fast | chars/tok PT | chars/tok EN | chars/tok code | "
         "Encode (M chr/s) | Batch×32 (M chr/s) | Decode (M tok/s) |")
    s = "|---|---:|:---:|---:|---:|---:|---:|---:|---:|"
    out = [h, s]
    for r in rows:
        out.append(
            f"| `{r.model.split('/')[-1]}` | {r.vocab} | {'✓' if r.is_fast else '✗'} | "
            f"{r.cpt_pt:.2f} | {r.cpt_en:.2f} | {r.cpt_code:.2f} | "
            f"{r.encode_chars_per_s/1e6:.2f} | {r.batch_chars_per_s/1e6:.2f} | "
            f"{r.decode_tokens_per_s/1e6:.2f} |"
        )
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", help="csv de modelos")
    ap.add_argument("--out", default=str(RESULTS_DIR / "tokenizer_results"))
    args = ap.parse_args()

    models = args.models.split(",") if args.models else DEFAULT_MODELS
    rows = [r for m in models if (r := bench(m)) is not None]

    if not rows:
        print("Sem resultados.")
        return

    md = to_markdown(rows)
    print("\n" + md)

    out_md = Path(f"{args.out}.md")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(md + "\n", encoding="utf-8")
    print(f"\n✔  Resultados em {out_md}")

    # Eficiência PT vs EN — útil para custos/contexto
    best_pt = max(rows, key=lambda r: r.cpt_pt)
    worst_pt = min(rows, key=lambda r: r.cpt_pt)
    print(f"\n🇵🇹 Mais eficiente em PT: {best_pt.model.split('/')[-1]} ({best_pt.cpt_pt:.2f} chars/tok)")
    print(f"🇵🇹 Menos eficiente em PT: {worst_pt.model.split('/')[-1]} ({worst_pt.cpt_pt:.2f} chars/tok)")
    print(f"   → Diferença: {(best_pt.cpt_pt/worst_pt.cpt_pt - 1)*100:+.0f}% menos tokens para o mesmo texto.")


if __name__ == "__main__":
    main()
