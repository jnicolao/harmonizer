"""Thin client for the local Ollama server."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

import httpx

from harmonizer.config import settings

EMBED_MODEL = "bge-m3"
CHAT_MODEL = "qwen3:8b"


def embed(texts: list[str], model: str = EMBED_MODEL) -> list[list[float]]:
    resp = httpx.post(
        f"{settings.ollama_host}/api/embed",
        json={"model": model, "input": texts},
        timeout=600,
    )
    resp.raise_for_status()
    return resp.json()["embeddings"]


def to_pgvector(vec: list[float]) -> str:
    """pgvector accepts its text form, so no extra adapter is needed."""
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"


@dataclass(frozen=True)
class ChatResult:
    data: dict
    prompt_tokens: int
    output_tokens: int
    seconds: float


def chat(
    system: str,
    user: str,
    schema: dict,
    model: str = CHAT_MODEL,
    num_ctx: int = 8192,
    num_predict: int = 600,
) -> ChatResult:
    """One structured-output call: thinking off, temperature 0, JSON matching `schema`."""
    started = time.perf_counter()
    resp = httpx.post(
        f"{settings.ollama_host}/api/chat",
        json={
            "model": model,
            "stream": False,
            "think": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "format": schema,
            "options": {"temperature": 0, "num_ctx": num_ctx, "num_predict": num_predict},
        },
        timeout=600,
    )
    resp.raise_for_status()
    body = resp.json()
    return ChatResult(
        data=json.loads(body["message"]["content"]),
        prompt_tokens=int(body.get("prompt_eval_count") or 0),
        output_tokens=int(body.get("eval_count") or 0),
        seconds=time.perf_counter() - started,
    )
