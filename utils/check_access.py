"""
Verifica acesso aos modelos HF antes de correr o benchmark.

Uso:
  python check_access.py                                  # usa lista por defeito
  python check_access.py meta-llama/Llama-3.2-1B-Instruct Qwen/Qwen2.5-0.5B-Instruct

Lê HF_TOKEN do ambiente ou de .env (se python-dotenv estiver instalado).
Sai com código 0 se todos os modelos estão acessíveis, 1 caso contrário.
"""

from __future__ import annotations

import os
import sys
from typing import Iterable

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from huggingface_hub import HfApi
from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError, HfHubHTTPError


DEFAULT_MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct",
]


def check_model(api: HfApi, model_id: str, token: str | None) -> tuple[bool, str]:
    """Devolve (ok, mensagem). Tenta ler ficheiro pequeno (config.json)."""
    try:
        api.model_info(model_id, token=token)
    except GatedRepoError:
        return False, "gated — pede acesso no HF e aceita os termos"
    except RepositoryNotFoundError:
        return False, "not found (404)"
    except HfHubHTTPError as e:
        status = getattr(e.response, "status_code", "?")
        if status == 401:
            return False, "401 — token inválido ou não fornecido"
        if status == 403:
            return False, "403 — sem permissão"
        return False, f"HTTP {status}: {e}"
    except Exception as e:  # pragma: no cover
        return False, f"{type(e).__name__}: {e}"

    # model_info pode passar mesmo para gated se metadata for público;
    # tenta também resolver um ficheiro real para confirmar download access
    try:
        api.hf_hub_download(model_id, "config.json", token=token, etag_timeout=10)
    except GatedRepoError:
        return False, "gated — metadata visível mas download bloqueado"
    except Exception:
        pass  # config.json pode não existir em alguns repos; basta model_info

    return True, "ok"


def main(models: Iterable[str]) -> int:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    api = HfApi()

    whoami = "(anonymous)"
    if token:
        try:
            info = api.whoami(token=token)
            whoami = info.get("name") or info.get("email") or "(unknown)"
        except Exception as e:
            whoami = f"(token inválido: {e})"

    print(f"== HF access check ==  user={whoami}  token={'set' if token else 'unset'}")

    all_ok = True
    for m in models:
        ok, msg = check_model(api, m, token)
        mark = "✔" if ok else "✗"
        print(f"  {mark}  {m:50s}  {msg}")
        all_ok &= ok

    return 0 if all_ok else 1


if __name__ == "__main__":
    args = sys.argv[1:] or DEFAULT_MODELS
    sys.exit(main(args))
