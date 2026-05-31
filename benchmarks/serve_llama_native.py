"""Servidor llama.cpp (`llama-server`) na Intel Arc GPU (SYCL), com *tool-calling*
nativo via `--jinja`.

Delegamos no servidor C++ oficial do llama.cpp: com `--jinja` ele usa o *chat
template* embebido no próprio GGUF (o template oficial do Qwen2.5, formato Hermes
`<tools>`/`<tool_call>`/`<tool_response>`) e tem um *parser* de *tool calls* nativo
(com gramática), por isso o opencode recebe `tool_calls` correctos sem código nosso
a manter.

Este *launcher* reaproveita o catálogo/escolha de modelo, a resolução do ficheiro
GGUF e o cálculo de contexto (serve_common) e depois faz `exec` do binário
`llama-server` com os argumentos certos. Compila o binário com:

    make setup-llama-server     # clona + compila llama.cpp (SYCL/Intel)

e arranca com:

    make serve-llama            # escolhe o modelo e serve
"""
from __future__ import annotations

import argparse
import multiprocessing
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from serve_common import (  # noqa: E402
    AUTO_CTX_CAP,
    RECOMMENDED_MODEL,
    _load_token,
    get_gguf_path,
    opencode_config,
    resolve_ctx,
    select_model_interactively,
)

DEFAULT_SERVER_BIN = os.path.expanduser(
    os.environ.get("LLAMA_CPP_DIR", "~/.cache/llama/llama.cpp")
) + "/build/bin/llama-server"


def find_server_bin(explicit: str | None) -> str:
    """Localiza o binário llama-server (CLI > env LLAMA_SERVER_BIN > build > PATH)."""
    candidates = [
        explicit,
        os.environ.get("LLAMA_SERVER_BIN"),
        DEFAULT_SERVER_BIN,
        shutil.which("llama-server"),
    ]
    for c in candidates:
        if c and os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    raise SystemExit(
        "Binário 'llama-server' não encontrado.\n"
        f"  Procurado em: {DEFAULT_SERVER_BIN} e no PATH.\n"
        "  Compila-o com:  make setup-llama-server\n"
        "  Ou aponta para ele com --server-bin / LLAMA_SERVER_BIN."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Servidor nativo llama.cpp (llama-server --jinja) na Intel Arc GPU"
    )
    parser.add_argument("--model", default=None,
                        help="Modelo HF a servir. Se omitido, pergunta "
                             "interactivamente (só os que cabem na memória).")
    parser.add_argument("--quant", default="q4_k_m",
                        help="Padrão de quantização (ex: q4_k_m, q8_0, f16)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--n-ctx", type=int, default=0,
                        help="Janela de contexto. 0 = automático (treino, limitado "
                             "por --max-ctx).")
    parser.add_argument("--max-ctx", type=int, default=AUTO_CTX_CAP,
                        help=f"Limite do modo automático (default: {AUTO_CTX_CAP}).")
    parser.add_argument("--n-gpu-layers", type=int, default=999,
                        help="Camadas na GPU (default: 999 = todas; 0 = CPU).")
    parser.add_argument("--n-threads", type=int, default=None,
                        help="Threads CPU (default: todos os cores)")
    parser.add_argument("--alias", default=None,
                        help="Nome do modelo exposto em /v1/models (default: --model)")
    parser.add_argument("--server-bin", default=None,
                        help="Caminho para o binário llama-server (override).")
    parser.add_argument("--no-jinja", action="store_true",
                        help="Desliga --jinja (sem tool-calling nativo).")
    parser.add_argument("--parallel", type=int, default=1,
                        help="Slots de inferência em paralelo (default: 1). "
                             "Com 1, um único cliente (opencode) usa toda a "
                             "janela de contexto; >1 divide --ctx-size pelos slots.")
    parser.add_argument("--chat-template-file", default=None,
                        help="Ficheiro Jinja de chat-template a usar em vez do "
                             "embutido no GGUF (override do --jinja).")
    parser.add_argument("--print-opencode-config", action="store_true",
                        help="Imprime a configuração do provider do opencode "
                             "(JSON, a partir do catálogo) e sai.")
    args = parser.parse_args()

    if args.print_opencode_config:
        import json
        cfg = opencode_config(base_url=f"http://{args.host}:{args.port}/v1")
        print(json.dumps(cfg, indent=2, ensure_ascii=False))
        return

    server_bin = find_server_bin(args.server_bin)
    n_threads = args.n_threads or multiprocessing.cpu_count()
    token = _load_token()

    model_id = args.model
    if model_id is None:
        plan_ctx = args.n_ctx if args.n_ctx > 0 else args.max_ctx
        model_id = select_model_interactively(plan_ctx)
        print()
    alias = args.alias or model_id

    print("== Servidor nativo llama.cpp (llama-server / SYCL / Intel Arc GPU) ==")
    print(f"Binário : {server_bin}")
    print(f"Modelo  : {model_id}  (quant={args.quant.upper()})")

    gguf_path = get_gguf_path(model_id, args.quant, token)
    size_mb = gguf_path.stat().st_size / 1024 ** 2
    print(f"Ficheiro: {gguf_path.name} ({size_mb:.0f} MB)")

    n_ctx, ctx_msg = resolve_ctx(model_id, gguf_path, args.n_ctx, args.max_ctx)
    print(f"Contexto: {ctx_msg}")
    print(f"Alias   : {alias}")
    print(f"Tools   : {'nativo (--jinja, template do GGUF)' if not args.no_jinja else 'desligado'}")
    print(f"Endpoint: http://{args.host}:{args.port}/v1")
    print()
    print("Configura o opencode com:")
    print(f"  baseURL = http://{args.host}:{args.port}/v1")
    print("  apiKey  = qualquer-valor (não é validado)")
    print(f"  model   = {alias}")
    print()
    if model_id == RECOMMENDED_MODEL:
        print(f"(modelo recomendado para opencode: {RECOMMENDED_MODEL})")
        print()
    elif "Coder" in model_id and not args.no_jinja:
        print("AVISO: modelos Qwen2.5-Coder embrulham as chamadas de ferramentas "
              "em ```json e o parser nativo do llama.cpp (peg-native) não as "
              f"extrai. Para tool-calling com o opencode usa {RECOMMENDED_MODEL}.")
        print()

    cmd = [
        server_bin,
        "--model", str(gguf_path),
        "--alias", alias,
        "--host", args.host,
        "--port", str(args.port),
        "--ctx-size", str(n_ctx),
        "--n-gpu-layers", str(args.n_gpu_layers),
        "--threads", str(n_threads),
        "--parallel", str(args.parallel),
    ]
    if not args.no_jinja:
        cmd.append("--jinja")
    if args.chat_template_file:
        cmd += ["--chat-template-file", args.chat_template_file]

    sys.stdout.flush()
    # exec: substitui este processo pelo llama-server (sinais/CTRL-C directos).
    os.execv(server_bin, cmd)


if __name__ == "__main__":
    main()
