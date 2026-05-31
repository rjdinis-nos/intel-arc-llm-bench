"""
Serve a GGUF model with llama-cpp-python's OpenAI-compatible server on the
Intel Arc GPU (SYCL), so editors/agents like opencode can use it as a local
provider.

The model is resolved (and downloaded if needed) exactly like the benchmark,
then exposed on http://HOST:PORT/v1 with all layers offloaded to the GPU.

Uso:
  python serve_llama.py
  python serve_llama.py --model Qwen/Qwen2.5-1.5B-Instruct --quant q4_k_m \
      --host 127.0.0.1 --port 8080

Aponta o opencode (ou qualquer cliente OpenAI) para:
  baseURL : http://127.0.0.1:8080/v1
  apiKey  : qualquer-valor (não é validado)
  model   : o alias mostrado em /v1/models (default: nome do modelo HF)
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import re
import sys
import time
import uuid

# Intel Arc 140T (Xe-LPG) + oneAPI 2026.0: the SYCL persistent device-code
# (JIT) cache segfaults while reading cached kernels from disk during
# inference. Disable it before the SYCL runtime initializes.
os.environ.setdefault("SYCL_CACHE_PERSISTENT", "0")

try:
    import uvicorn
    from llama_cpp.server.app import create_app
    from llama_cpp.server.settings import ModelSettings, ServerSettings
    from llama_cpp.llama_chat_format import register_chat_completion_handler
except ImportError as exc:  # pragma: no cover - depends on optional extras
    raise SystemExit(
        f"Dependências em falta ({exc.name}).\n"
        "O servidor precisa do llama-cpp-python (build SYCL) e dos extras de "
        "servidor:\n"
        "  make setup-llama   # compila o motor para a GPU (requer oneAPI)\n"
        "  uv pip install 'uvicorn' 'fastapi' 'pydantic-settings' "
        "'sse-starlette' 'starlette-context' 'PyYAML'"
    )

from serve_common import (
    AUTO_CTX_CAP,
    _load_token,
    get_gguf_path,
    resolve_ctx,
    select_model_interactively,
)


# Limite de contexto em modo automático e catálogo de modelos vivem em
# serve_common.py (partilhados com o servidor nativo). Importados acima.


def resolve_chat_format(model_id: str) -> str | None:
    """Escolhe o handler de chat para tool-calling estilo OpenAI.

    O servidor llama-cpp, sem handler de funções, devolve as tool calls como
    texto no `content` (não as parseia para o array `tool_calls`). O handler
    embutido 'chatml-function-calling' parseia-as mas (1) rebenta em
    stream+tool_choice=auto e (2) usa um formato genérico que o Qwen não viu no
    treino (o modelo entra em loop ou vaza 'functions.x:' no texto). Por isso os
    modelos Qwen2.5 usam um handler nativo (QWEN_TOOL_HANDLER) que fala o formato
    Hermes <tools>/<tool_call>/<tool_response> em que o Qwen foi treinado. Para
    outras famílias devolvemos None (o llama.cpp usa o template do GGUF).
    """
    if "Qwen" in model_id:
        return QWEN_TOOL_HANDLER
    return None


# Handler de chat nativo do Qwen2.5 registado abaixo, com tool-calling estilo
# OpenAI e suporte a streaming. Substitui o 'chatml-function-calling' embutido,
# que rebentava com "Automatic streaming tool choice is not supported" (o
# opencode pede stream=true + tool_choice=auto → o cliente via "ASGI callable
# returned without completing response") e cujo formato genérico fazia o Qwen
# entrar em loop a repetir tool calls (não via os resultados) ou vazar a sintaxe
# das funções para o conteúdo.
QWEN_TOOL_HANDLER = "qwen2.5-tool-calling"

_QWEN_TOOL_PREAMBLE = (
    "# Tools\n\nYou may call one or more functions to assist with the user "
    "query.\n\nYou are provided with function signatures within "
    "<tools></tools> XML tags:\n<tools>\n{tools}\n</tools>\n\nFor each function "
    "call, return a json object with function name and arguments within "
    "<tool_call></tool_call> XML tags:\n<tool_call>\n{{\"name\": "
    "<function-name>, \"arguments\": <args-json-object>}}\n</tool_call>"
)

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)

# Parâmetros de geração que aceitamos repassar a llama.create_completion.
_GEN_PARAM_KEYS = (
    "temperature", "top_p", "top_k", "min_p", "typical_p", "presence_penalty",
    "frequency_penalty", "repeat_penalty", "tfs_z", "mirostat_mode",
    "mirostat_tau", "mirostat_eta", "seed", "logits_processor", "logit_bias",
)


def _render_qwen_prompt(messages, tools):
    """Renderiza a conversa no formato ChatML de tool-calling do Qwen2.5."""
    system_content = ""
    body = []
    for msg in messages:
        if msg.get("role") == "system" and not body and not system_content:
            system_content = msg.get("content") or ""
        else:
            body.append(msg)

    if tools:
        tool_lines = "\n".join(json.dumps(t) for t in tools)
        preamble = _QWEN_TOOL_PREAMBLE.format(tools=tool_lines)
        system_content = (
            f"{system_content}\n\n{preamble}" if system_content else preamble
        )

    parts = [f"<|im_start|>system\n{system_content}<|im_end|>\n"]

    pending_tool_responses: list[str] = []

    def flush_tool_responses():
        if pending_tool_responses:
            joined = "\n".join(
                f"<tool_response>\n{c}\n</tool_response>"
                for c in pending_tool_responses
            )
            parts.append(f"<|im_start|>user\n{joined}<|im_end|>\n")
            pending_tool_responses.clear()

    for msg in body:
        role = msg.get("role")
        if role == "tool":
            pending_tool_responses.append(str(msg.get("content") or ""))
            continue
        flush_tool_responses()
        if role == "user":
            parts.append(f"<|im_start|>user\n{msg.get('content') or ''}<|im_end|>\n")
        elif role == "system":
            parts.append(f"<|im_start|>system\n{msg.get('content') or ''}<|im_end|>\n")
        elif role == "assistant":
            content = msg.get("content") or ""
            block = content
            for tc in msg.get("tool_calls") or []:
                fn = tc["function"]
                args = fn["arguments"]
                if not isinstance(args, str):
                    args = json.dumps(args)
                call = json.dumps({"name": fn["name"]})[:-1]
                block += f"\n<tool_call>\n{call}, \"arguments\": {args}}}\n</tool_call>"
            parts.append(f"<|im_start|>assistant\n{block}<|im_end|>\n")
    flush_tool_responses()

    parts.append("<|im_start|>assistant\n")
    return "".join(parts)


def _make_tool_call(obj):
    """Constrói um tool_call OpenAI a partir de {"name":..,"arguments":..}."""
    name = obj.get("name")
    if not name:
        return None
    arguments = obj.get("arguments", {})
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments)
    return {
        "id": f"call_{uuid.uuid4().hex[:24]}",
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def _parse_qwen_tool_calls(text, tool_names=None):
    """Extrai tool calls do texto gerado → (content, tool_calls).

    Caminho principal: blocos `<tool_call>{...}</tool_call>` (formato nativo do
    Qwen). Fallback para modelos mais fracos que ignoram as instruções e emitem
    a chamada como JSON em bloco markdown (```json {...} ```) ou JSON puro: só é
    tratada como tool call se tiver `name` (de uma das ferramentas disponíveis)
    e `arguments`, para não confundir JSON legítimo no conteúdo.
    """
    tool_calls = []
    matches = list(_TOOL_CALL_RE.finditer(text))
    if matches:
        for match in matches:
            try:
                obj = json.loads(match.group(1).strip())
            except Exception:
                continue
            tc = _make_tool_call(obj)
            if tc:
                tool_calls.append(tc)
        if tool_calls:
            return _TOOL_CALL_RE.sub("", text).strip(), tool_calls

    candidates = [(m.start(), m.end(), m.group(1)) for m in _FENCED_JSON_RE.finditer(text)]
    if not candidates:
        stripped = text.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            candidates = [(0, len(text), stripped)]
    spans = []
    for start, end, raw in candidates:
        try:
            obj = json.loads(raw.strip())
        except Exception:
            continue
        if not (isinstance(obj, dict) and "arguments" in obj):
            continue
        if tool_names is not None and obj.get("name") not in tool_names:
            continue
        tc = _make_tool_call(obj)
        if tc:
            tool_calls.append(tc)
            spans.append((start, end))
    content = text
    for start, end in sorted(spans, reverse=True):
        content = content[:start] + content[end:]
    return content.strip(), tool_calls


def _gen_kwargs(kwargs):
    out = {k: kwargs[k] for k in _GEN_PARAM_KEYS if kwargs.get(k) is not None}
    if kwargs.get("max_tokens") is not None:
        out["max_tokens"] = kwargs["max_tokens"]
    return out


def _stream_text_as_chat_chunks(llama, prompt, stop, gen_kwargs, model):
    """Gera em streaming (sem tools) e converte cada chunk de texto em chat chunk."""
    base = {
        "id": "chatcmpl-" + uuid.uuid4().hex,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
    }

    def chunk(delta, finish_reason=None):
        return {**base, "choices": [{
            "index": 0, "delta": delta,
            "finish_reason": finish_reason, "logprobs": None,
        }]}

    yield chunk({"role": "assistant"})
    finish = "stop"
    for piece in llama.create_completion(
        prompt=prompt, stop=stop, stream=True, **gen_kwargs
    ):
        ch = piece["choices"][0]
        text = ch.get("text", "")
        if text:
            yield chunk({"content": text})
        if ch.get("finish_reason"):
            finish = ch["finish_reason"]
    yield chunk({}, finish_reason=finish)


@register_chat_completion_handler(QWEN_TOOL_HANDLER)
def qwen_tool_calling(llama, **kwargs):
    """Handler de chat nativo do Qwen2.5 com tool-calling e streaming.

    Renderiza a conversa no formato Hermes do Qwen, gera, e parseia os blocos
    <tool_call> de volta para o array `tool_calls` da API OpenAI. Quando há tools
    geramos sem streaming (para poder parsear) e reemitimos como chunks; sem
    tools fazemos streaming token-a-token normal.
    """
    messages = kwargs.get("messages") or []
    tools = kwargs.get("tools")
    tool_choice = kwargs.get("tool_choice")
    if tool_choice == "none":
        tools = None
    stream = kwargs.get("stream", False)
    model = kwargs.get("model") or "qwen"

    prompt = _render_qwen_prompt(messages, tools)
    stop = ["<|im_end|>", "<|im_start|>"]
    gen_kwargs = _gen_kwargs(kwargs)

    if stream and not tools:
        return _stream_text_as_chat_chunks(llama, prompt, stop, gen_kwargs, model)

    completion = llama.create_completion(
        prompt=prompt, stop=stop, stream=False, **gen_kwargs
    )
    text = completion["choices"][0]["text"]
    tool_names = None
    if tools:
        tool_names = {t["function"]["name"] for t in tools}
    content, tool_calls = _parse_qwen_tool_calls(text, tool_names)

    if os.environ.get("SERVE_DUMP_REQUESTS"):
        try:
            with open(os.environ["SERVE_DUMP_REQUESTS"], "a") as fh:
                fh.write(f"\n---- MODEL OUTPUT (tools={bool(tools)}, "
                         f"ncalls={len(tool_calls)}) ----\n{text}\n")
        except Exception:
            pass

    message: dict = {"role": "assistant", "content": content or None}
    if tool_calls:
        message["tool_calls"] = tool_calls
        message["content"] = None

    response = {
        "id": "chatcmpl-" + uuid.uuid4().hex,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": message,
            "logprobs": None,
            "finish_reason": "tool_calls" if tool_calls else "stop",
        }],
        "usage": completion.get("usage", {}),
    }

    if stream:
        return _chat_completion_to_stream_chunks(response)
    return response


def _chat_completion_to_stream_chunks(response):
    """Converte uma resposta de chat (não-stream) em chunks de streaming OpenAI."""
    choice = response["choices"][0]
    message = choice.get("message", {})
    base = {
        "id": response["id"],
        "object": "chat.completion.chunk",
        "created": response["created"],
        "model": response["model"],
    }

    def chunk(delta, finish_reason=None):
        return {**base, "choices": [{
            "index": 0, "delta": delta,
            "finish_reason": finish_reason, "logprobs": None,
        }]}

    yield chunk({"role": "assistant"})
    tool_calls = message.get("tool_calls")
    if tool_calls:
        for i, tc in enumerate(tool_calls):
            yield chunk({"tool_calls": [{
                "index": i,
                "id": tc.get("id"),
                "type": "function",
                "function": {
                    "name": tc["function"]["name"],
                    "arguments": tc["function"]["arguments"],
                },
            }]})
    elif message.get("content"):
        yield chunk({"content": message["content"]})
    yield chunk({}, finish_reason=choice.get("finish_reason", "stop"))


class NullContentFixMiddleware:
    """Corrige pedidos de chat com `content: null` em mensagens `assistant`.

    O schema de pedido do servidor llama-cpp (0.3.x) declara o `content` da
    mensagem assistant como string e rejeita `null` com um erro de validação,
    mesmo quando a mensagem traz `tool_calls` (caso válido na spec OpenAI). É
    exactamente o que o opencode/ai-sdk envia no turno seguinte a uma tool call,
    o que fazia o pedido rebentar ("ASGI callable returned without completing
    response"). Este middleware ASGI intercepta o corpo do POST /chat/completions
    e troca esses `null` por "" antes da validação.
    """

    def __init__(self, app):
        self.app = app

    @staticmethod
    def _patch_body(body: bytes) -> bytes:
        try:
            data = json.loads(body)
        except Exception:
            return body
        messages = data.get("messages")
        if not isinstance(messages, list):
            return body
        changed = False
        for msg in messages:
            if (isinstance(msg, dict) and msg.get("role") == "assistant"
                    and "content" in msg and msg["content"] is None):
                msg["content"] = ""
                changed = True
        if not changed:
            return body
        return json.dumps(data).encode("utf-8")

    async def __call__(self, scope, receive, send):
        if (scope.get("type") != "http" or scope.get("method") != "POST"
                or not scope.get("path", "").endswith("/chat/completions")):
            await self.app(scope, receive, send)
            return

        body = b""
        more_body = True
        while more_body:
            message = await receive()
            if message["type"] != "http.request":
                more_body = False
                break
            body += message.get("body", b"")
            more_body = message.get("more_body", False)

        body = self._patch_body(body)

        if os.environ.get("SERVE_DUMP_REQUESTS"):
            try:
                dump_path = os.environ["SERVE_DUMP_REQUESTS"]
                with open(dump_path, "ab") as fh:
                    fh.write(b"\n==== REQUEST ====\n" + body + b"\n")
            except Exception:
                pass

        sent = False

        async def patched_receive():
            nonlocal sent
            if not sent:
                sent = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        await self.app(scope, patched_receive, send)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Servidor OpenAI-compatível (llama.cpp / SYCL / Intel Arc GPU)"
    )
    parser.add_argument("--model", default=None,
                        help="Modelo HF a servir. Se omitido, pergunta "
                             "interactivamente (só os que cabem na memória).")
    parser.add_argument("--quant", default="q4_k_m",
                        help="Padrão de quantização (ex: q4_k_m, q8_0, f16)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--n-ctx", type=int, default=0,
                        help="Janela de contexto. 0 = automático: usa o contexto "
                             f"de treino do modelo, limitado a --max-ctx.")
    parser.add_argument("--max-ctx", type=int, default=AUTO_CTX_CAP,
                        help=f"Limite do modo automático (default: {AUTO_CTX_CAP}).")
    parser.add_argument("--n-threads", type=int, default=None,
                        help="Threads CPU (default: todos os cores)")
    parser.add_argument("--alias", default=None,
                        help="Nome do modelo exposto em /v1/models "
                             "(default: o --model)")
    parser.add_argument("--chat-format", default="auto",
                        help="Handler de chat do llama-cpp. 'auto' escolhe "
                             "'chatml-function-calling' para modelos Qwen "
                             "(tool-calling p/ opencode); 'none' usa o template "
                             "do GGUF. Ou um nome explícito (ex: chatml).")
    args = parser.parse_args()

    n_threads = args.n_threads or multiprocessing.cpu_count()
    token = _load_token()

    # Sem --model: perguntar interactivamente, mostrando só os que cabem.
    # A KV-cache é dimensionada pelo pior caso (contexto fixo ou --max-ctx).
    model_id = args.model
    if model_id is None:
        plan_ctx = args.n_ctx if args.n_ctx > 0 else args.max_ctx
        model_id = select_model_interactively(plan_ctx)
        print()

    alias = args.alias or model_id

    print("== Servidor llama.cpp (SYCL/Intel Arc GPU) ==")
    print(f"Modelo  : {model_id}  (quant={args.quant.upper()})")

    gguf_path = get_gguf_path(model_id, args.quant, token)
    size_mb = gguf_path.stat().st_size / 1024 ** 2
    print(f"Ficheiro: {gguf_path.name} ({size_mb:.0f} MB)")

    n_ctx, ctx_msg = resolve_ctx(model_id, gguf_path, args.n_ctx, args.max_ctx)
    print(f"Contexto: {ctx_msg}")

    print(f"Alias   : {alias}")

    if args.chat_format == "auto":
        chat_format = resolve_chat_format(model_id)
    elif args.chat_format.lower() == "none":
        chat_format = None
    else:
        chat_format = args.chat_format
    print(f"Chat fmt: {chat_format or 'template do GGUF'}")

    print(f"Endpoint: http://{args.host}:{args.port}/v1")
    print()
    print("Configura o opencode com:")
    print(f"  baseURL = http://{args.host}:{args.port}/v1")
    print("  apiKey  = qualquer-valor (não é validado)")
    print(f"  model   = {alias}")
    print()

    server_settings = ServerSettings(host=args.host, port=args.port)
    model_settings = [
        ModelSettings(
            model=str(gguf_path),
            model_alias=alias,
            n_gpu_layers=-1,
            n_threads=n_threads,
            n_ctx=n_ctx,
            # logits_all=True (o default do servidor) força o cálculo de logits
            # para TODOS os tokens, alocando um tensor n_ctx × n_vocab. Em modelos
            # grandes (7B, vocab ~152k) isso são vários GB numa única alocação na
            # iGPU Arc e rebenta com UR_RESULT_ERROR_DEVICE_LOST. Clientes de chat
            # (opencode) só precisam dos logits do último token.
            logits_all=False,
            chat_format=chat_format,
            verbose=False,
        )
    ]

    app = create_app(
        server_settings=server_settings,
        model_settings=model_settings,
    )
    app = NullContentFixMiddleware(app)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
