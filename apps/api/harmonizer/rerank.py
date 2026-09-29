"""Cross-encoder reranking with bge-reranker-v2-m3, run locally.

The weights download once from the Hugging Face Hub into data/hf and load on
Apple's MPS backend in half precision when available, otherwise CPU.

Measured on this task the general-purpose reranker does not beat fused
retrieval in any configuration (see `hz eval rerank-ablation`), so it is
opt-in and off by default.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from harmonizer.retrieval import Candidate

REPO_ROOT = Path(__file__).resolve().parents[3]
os.environ.setdefault("HF_HOME", str(REPO_ROOT / "data" / "hf"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

RERANK_MODEL = "BAAI/bge-reranker-v2-m3"


class Reranker:
    def __init__(self, model_name: str = RERANK_MODEL, max_length: int = 512) -> None:
        import torch
        from sentence_transformers import CrossEncoder

        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        kwargs = {"torch_dtype": torch.float16} if self.device == "mps" else {}
        self.model = CrossEncoder(
            model_name, max_length=max_length, device=self.device, model_kwargs=kwargs
        )
        self.model_name = model_name
        self.max_length = max_length

    def score(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        scores = self.model.predict(pairs, batch_size=32, show_progress_bar=False)
        return np.asarray(scores, dtype=np.float64)

    def rerank(self, query: str, candidates: list[Candidate], k: int) -> list[Candidate]:
        if not candidates:
            return []
        order = np.argsort(-self.score([(query, c.text) for c in candidates]))
        return [candidates[i] for i in order[:k]]
