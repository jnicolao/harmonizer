"""Turn a ranked list of precedent candidates into a code prediction.

nearest        the code of the top candidate, the baseline so far
vote           every distinct code scored by the summed reciprocal rank of the
               neighbours that carry it, over the top k
hierarchical   the same vote settled one level at a time: chapter first, then
               heading within that chapter, then subheading, and so on, so a
               few strong neighbours cannot be outvoted by scattered ones
"""

from collections import defaultdict

from harmonizer.retrieval import Candidate

LEVELS = (2, 4, 6, 8, 10)


def _weighted(cands: list[Candidate], k: int) -> list[tuple[str, float]]:
    return [(c.code, 1.0 / (rank + 1)) for rank, c in enumerate(cands[:k])]


def nearest(cands: list[Candidate]) -> list[str]:
    seen: list[str] = []
    for c in cands:
        if c.code not in seen:
            seen.append(c.code)
    return seen


def vote(cands: list[Candidate], k: int) -> list[str]:
    scores: dict[str, float] = defaultdict(float)
    for code, w in _weighted(cands, k):
        scores[code] += w
    return sorted(scores, key=lambda code: -scores[code])


def hierarchical(cands: list[Candidate], k: int) -> list[str]:
    pool = _weighted(cands, k)
    prefix = ""
    for level in LEVELS:
        scores: dict[str, float] = defaultdict(float)
        for code, w in pool:
            if len(code) >= level:
                scores[code[:level]] += w
        if not scores:
            break
        prefix = max(scores, key=lambda p: scores[p])
        pool = [(code, w) for code, w in pool if code.startswith(prefix)]
    rest = [code for code in vote(cands, k) if code != prefix]
    return [prefix, *rest] if prefix else rest
