"""
Benchmark de débito (tokens/segundo) de um LLM local via HuggingFace Transformers.

Mede:
  - Prefill (prompt) tokens/s
  - Decode (generation) tokens/s
  - Latência ao primeiro token (TTFT)
  - Latência total

Uso:
  python benchmark_tps.py
  python benchmark_tps.py --model Qwen/Qwen2.5-0.5B-Instruct --new-tokens 256 --runs 5
"""

from __future__ import annotations

import argparse
import gc
import time
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread


DEFAULT_PROMPT = (
    "Explica de forma clara e detalhada o que é a inferência de um modelo de "
    "linguagem grande, incluindo as fases de prefill e decode, e porque é que "
    "o débito (tokens por segundo) é uma métrica importante."
)


def pick_device() -> str:
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def sync(device: str) -> None:
    if device == "xpu":
        torch.xpu.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


@dataclass
class RunResult:
    prompt_tokens: int
    generated_tokens: int
    ttft_s: float
    total_s: float

    @property
    def decode_s(self) -> float:
        return max(self.total_s - self.ttft_s, 1e-9)

    @property
    def prefill_tps(self) -> float:
        return self.prompt_tokens / max(self.ttft_s, 1e-9)

    @property
    def decode_tps(self) -> float:
        # tokens gerados após o primeiro
        return max(self.generated_tokens - 1, 0) / self.decode_s

    @property
    def overall_tps(self) -> float:
        return self.generated_tokens / max(self.total_s, 1e-9)


def run_once(model, tokenizer, device: str, prompt: str, new_tokens: int) -> RunResult:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_tokens = inputs.input_ids.shape[-1]

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    gen_kwargs = dict(
        **inputs,
        max_new_tokens=new_tokens,
        do_sample=False,
        streamer=streamer,
        pad_token_id=tokenizer.eos_token_id,
    )

    sync(device)
    t_start = time.perf_counter()

    thread = Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    ttft = None
    generated_text = ""
    for chunk in streamer:
        if ttft is None:
            ttft = time.perf_counter() - t_start
        generated_text += chunk

    thread.join()
    sync(device)
    total = time.perf_counter() - t_start

    generated_tokens = len(tokenizer(generated_text, add_special_tokens=False).input_ids)

    return RunResult(
        prompt_tokens=prompt_tokens,
        generated_tokens=generated_tokens,
        ttft_s=ttft if ttft is not None else total,
        total_s=total,
    )


def fmt(r: RunResult) -> str:
    return (
        f"prompt={r.prompt_tokens:>4}t  gen={r.generated_tokens:>4}t  "
        f"TTFT={r.ttft_s*1000:7.1f}ms  total={r.total_s:6.2f}s  "
        f"prefill={r.prefill_tps:7.2f} t/s  decode={r.decode_tps:7.2f} t/s  "
        f"overall={r.overall_tps:7.2f} t/s"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark tokens/s de um LLM local")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--new-tokens", type=int, default=256)
    parser.add_argument("--runs", type=int, default=3, help="número de execuções medidas")
    parser.add_argument("--warmup", type=int, default=1, help="execuções de warmup (não contabilizadas)")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--device", default=None, help="forçar device (xpu/cuda/cpu)")
    args = parser.parse_args()

    device = args.device or pick_device()
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]

    print(f"== Benchmark LLM ==")
    print(f"Modelo : {args.model}")
    print(f"Device : {device}")
    print(f"Dtype  : {args.dtype}")
    print(f"Runs   : {args.runs} (warmup={args.warmup})")
    print(f"Tokens : max_new_tokens={args.new_tokens}")
    print()

    print("A carregar tokenizer e modelo...")
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device)
    model.eval()
    print(f"Carregado em {time.perf_counter() - t0:.2f}s\n")

    # Aplica chat template se disponível
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": args.prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        prompt = args.prompt

    with torch.inference_mode():
        for i in range(args.warmup):
            print(f"[warmup {i+1}/{args.warmup}] ", end="", flush=True)
            r = run_once(model, tokenizer, device, prompt, args.new_tokens)
            print(fmt(r))

        results: list[RunResult] = []
        for i in range(args.runs):
            print(f"[run    {i+1}/{args.runs}] ", end="", flush=True)
            r = run_once(model, tokenizer, device, prompt, args.new_tokens)
            results.append(r)
            print(fmt(r))

    if results:
        n = len(results)
        avg_prefill = sum(r.prefill_tps for r in results) / n
        avg_decode = sum(r.decode_tps for r in results) / n
        avg_overall = sum(r.overall_tps for r in results) / n
        avg_ttft = sum(r.ttft_s for r in results) / n
        print()
        print(f"== Médias ({n} runs) ==")
        print(f"TTFT médio   : {avg_ttft*1000:.1f} ms")
        print(f"Prefill t/s  : {avg_prefill:.2f}")
        print(f"Decode  t/s  : {avg_decode:.2f}")
        print(f"Overall t/s  : {avg_overall:.2f}")

    del model
    gc.collect()
    if device == "xpu":
        torch.xpu.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
