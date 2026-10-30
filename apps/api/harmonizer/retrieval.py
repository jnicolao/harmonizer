"""Candidate retrieval over precedents (past rulings) and ten-digit tariff lines.

Three modes share one interface:
  dense   bge-m3 vectors in pgvector
  bm25    rank_bm25 over the same texts, held in memory (about 50k short docs)
  hybrid  reciprocal rank fusion of the two
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

import numpy as np
from rank_bm25 import BM25Okapi

TOKEN = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset(
    "the a an and or of for to in on with by is are be as at from this that these those it its "
    "which such other than not was were will been has have had into under over per each".split()
)
RRF_K = 60
FUSE_DEPTH = 100


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN.findall(text.lower()) if len(t) > 2 and t not in STOPWORDS]


@dataclass(frozen=True)
class Candidate:
    key: str  # ruling number, or htsno with dots
    code: str  # tariff code, digits only
    text: str
    ruling_date: date | None = None


def _rrf(*ranked: list[Candidate], k: int) -> list[Candidate]:
    scores: dict[str, float] = {}
    by_key: dict[str, Candidate] = {}
    for lst in ranked:
        for rank, cand in enumerate(lst):
            scores[cand.key] = scores.get(cand.key, 0.0) + 1.0 / (RRF_K + rank + 1)
            by_key[cand.key] = cand
    order = sorted(scores, key=lambda key: scores[key], reverse=True)
    return [by_key[key] for key in order[:k]]


class Retriever:
    def __init__(self, conn) -> None:
        self.conn = conn
        rows = conn.execute(
            "SELECT ruling_number, tariff_codes[1], query_text, ruling_date FROM cross_ruling "
            "WHERE embedding IS NOT NULL AND cardinality(tariff_codes) = 1"
        ).fetchall()
        self._precedent_docs = [Candidate(n, c, t, d) for n, c, t, d in rows]
        self._precedent_by_key = {c.key: c for c in self._precedent_docs}
        self._precedent_bm25 = BM25Okapi([tokenize(c.text) for c in self._precedent_docs])

        rows = conn.execute(
            "SELECT htsno, full_path FROM hts_line WHERE digits = 10 AND embedding IS NOT NULL"
        ).fetchall()
        self._line_docs = [Candidate(h, h.replace(".", ""), p) for h, p in rows]
        self._line_by_key = {c.key: c for c in self._line_docs}
        self._line_bm25 = BM25Okapi([tokenize(c.text) for c in self._line_docs])

    # precedents -------------------------------------------------------------

    def _precedents_dense(
        self, emb: str, k: int, before: date | None, exclude: str | None
    ) -> list[Candidate]:
        rows = self.conn.execute(
            "SELECT ruling_number FROM cross_ruling "
            "WHERE embedding IS NOT NULL AND cardinality(tariff_codes) = 1 "
            "  AND ruling_number <> %s AND (%s::date IS NULL OR ruling_date < %s) "
            "ORDER BY embedding <=> %s::vector LIMIT %s",
            (exclude or "", before, before, emb, k),
        ).fetchall()
        return [self._precedent_by_key[r[0]] for r in rows if r[0] in self._precedent_by_key]

    def _precedents_bm25(
        self, text: str, k: int, before: date | None, exclude: str | None
    ) -> list[Candidate]:
        scores = self._precedent_bm25.get_scores(tokenize(text))
        out: list[Candidate] = []
        for i in np.argsort(-scores):
            if scores[i] <= 0:
                break
            c = self._precedent_docs[i]
            if c.key == exclude or (before and c.ruling_date and c.ruling_date >= before):
                continue
            out.append(c)
            if len(out) == k:
                break
        return out

    def precedents(
        self,
        text: str,
        emb: str,
        k: int,
        mode: str = "hybrid",
        before: date | None = None,
        exclude: str | None = None,
    ) -> list[Candidate]:
        """Past rulings most like this product. `before`/`exclude` keep evals leak-free."""
        if mode == "dense":
            return self._precedents_dense(emb, k, before, exclude)
        if mode == "bm25":
            return self._precedents_bm25(text, k, before, exclude)
        depth = max(k, FUSE_DEPTH)
        return _rrf(
            self._precedents_dense(emb, depth, before, exclude),
            self._precedents_bm25(text, depth, before, exclude),
            k=k,
        )

    # tariff lines -----------------------------------------------------------

    def _lines_dense(self, emb: str, k: int) -> list[Candidate]:
        rows = self.conn.execute(
            "SELECT htsno FROM hts_line WHERE digits = 10 AND embedding IS NOT NULL "
            "ORDER BY embedding <=> %s::vector LIMIT %s",
            (emb, k),
        ).fetchall()
        return [self._line_by_key[r[0]] for r in rows]

    def _lines_bm25(self, text: str, k: int) -> list[Candidate]:
        scores = self._line_bm25.get_scores(tokenize(text))
        top = np.argsort(-scores)[:k]
        return [self._line_docs[i] for i in top if scores[i] > 0]

    def lines(self, text: str, emb: str, k: int, mode: str = "hybrid") -> list[Candidate]:
        """Ten-digit tariff lines most like this product."""
        if mode == "dense":
            return self._lines_dense(emb, k)
        if mode == "bm25":
            return self._lines_bm25(text, k)
        depth = max(k, FUSE_DEPTH)
        return _rrf(self._lines_dense(emb, depth), self._lines_bm25(text, depth), k=k)
