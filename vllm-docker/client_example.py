#!/usr/bin/env python3
"""
Simple vLLM client example — OpenAI-compatible API.

Reads MODEL_NAME / VLLM_PORT from the environment (or .env defaults) so it
matches whatever the server is serving.

Usage:
    python client_example.py
"""

import os

from openai import OpenAI

PORT = os.environ.get("VLLM_PORT", "8000")
MODEL = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")

# Point to local vLLM server
client = OpenAI(
    api_key="not-needed",
    base_url=f"http://localhost:{PORT}/v1",
)


def chat(message: str) -> str:
    """Send a chat message and get response."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "user", "content": message}
        ],
        max_tokens=256,
        temperature=0.7,
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    print(f"Connecting to vLLM at http://localhost:{PORT} (model: {MODEL})...\n")

    prompt = "Explain what is a Large Language Model in one sentence."
    print(f"Q: {prompt}")
    print()

    try:
        response = chat(prompt)
        print(f"A: {response}")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nMake sure the vLLM server is running:")
        print("  make start")

