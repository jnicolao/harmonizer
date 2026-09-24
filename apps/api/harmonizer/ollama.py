"""Thin client for the local Ollama server."""

import httpx

from harmonizer.config import settings

EMBED_MODEL = "bge-m3"


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
