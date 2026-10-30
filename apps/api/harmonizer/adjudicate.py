"""The adjudicator: one LLM call that argues for a code among retrieved candidates.

Input is the product description, the fused top-20 past rulings, the fused
top-10 tariff lines, and the hierarchical vote as a prior. Output is a code
from the candidate set, the ruling it leans on, a confidence and a rationale.
An answer outside the candidate set falls back to the vote and is flagged.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from harmonizer.ollama import ChatResult, chat
from harmonizer.retrieval import Candidate, Retriever
from harmonizer.vote import hierarchical

PRECEDENTS = 20
LINES = 10
VOTE_K = 20

SCHEMA = {
    "type": "object",
    "properties": {
        "code": {"type": "string", "description": "A tariff code copied from the candidates"},
        "precedent": {"type": "string", "description": "Ruling number relied on, or 'none'"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "rationale": {"type": "string"},
    },
    "required": ["code", "precedent", "confidence", "rationale"],
}

SYSTEM = """You are a customs classification specialist working with the Harmonized Tariff Schedule of the United States. You choose the tariff code for a product by comparing it with past binding rulings and with the text of the tariff lines.

Rules you follow:
- Pick exactly one code, copied verbatim from the candidates you are given. Never invent or edit a code.
- Classify by what the product is, what it is made of and what it is used for. The importer, brand and country of origin do not matter.
- Apply the General Rules of Interpretation: a specific description beats a general one; composite goods and sets go by the component that gives them their essential character; when two headings still both apply, take the one that occurs last in numerical order.
- Prefer a past ruling about the same kind of product over a tariff line that merely sounds similar. Prefer recent rulings over old ones when they disagree.
- Set confidence low when the candidates do not describe this product well."""


def dotted(code: str) -> str:
    parts = [code[:4], code[4:6], code[6:8], code[8:10]]
    return ".".join(p for p in parts if p)


def digits(code: str) -> str:
    return "".join(ch for ch in code if ch.isdigit())


def _line_text(path: str, limit: int = 240) -> str:
    if len(path) <= limit:
        return path
    return path[:80] + " … " + path[-(limit - 84) :]


def build_prompt(
    description: str, voted: str | None, precedents: list[Candidate], lines: list[Candidate]
) -> str:
    parts = ["PRODUCT", description.strip(), ""]
    if voted:
        parts += [
            "NEIGHBOUR VOTE",
            f"Weighted by similarity, the nearest past rulings favour chapter {voted[:2]} "
            f"and, settled level by level, code {dotted(voted)}. Treat this as a prior, not an answer.",
            "",
        ]
    parts.append("PAST RULINGS, nearest first")
    for c in precedents:
        parts.append(f"- [{c.key}] {dotted(c.code)}: {c.text[:240]}")
    parts += ["", "TARIFF LINES, nearest first"]
    for c in lines:
        parts.append(f"- {dotted(c.code)}: {_line_text(c.text)}")
    parts += [
        "",
        "TASK",
        "Choose the single best code for the PRODUCT from the candidates above. "
        "Cite the ruling number that supports it most directly, or 'none'. "
        "Give a confidence between 0 and 1. "
        "In the rationale, name the material, the function and the rule you applied, in two or three sentences.",
    ]
    return "\n".join(parts)


@dataclass(frozen=True)
class Decision:
    code: str
    precedent: str
    confidence: float
    rationale: str
    voted: str | None
    answered: str | None  # what the model actually said, digits only
    in_candidates: bool
    fallback: bool
    error: str | None
    prompt_tokens: int
    output_tokens: int
    seconds: float


def adjudicate(
    retriever: Retriever,
    text: str,
    emb: str,
    before: date | None = None,
    exclude: str | None = None,
) -> Decision:
    precedents = retriever.precedents(
        text, emb, PRECEDENTS, "hybrid", before=before, exclude=exclude
    )
    lines = retriever.lines(text, emb, LINES, "hybrid")
    votes = hierarchical(precedents, VOTE_K)
    voted = votes[0] if votes else None
    candidates = {c.code for c in precedents} | {c.code for c in lines}

    try:
        result: ChatResult = chat(SYSTEM, build_prompt(text, voted, precedents, lines), SCHEMA)
    except Exception as exc:  # noqa: BLE001 - any failure falls back to the vote
        return Decision(
            voted or "",
            "none",
            0.0,
            "",
            voted,
            None,
            False,
            True,
            f"{type(exc).__name__}: {exc}",
            0,
            0,
            0.0,
        )

    answered = digits(str(result.data.get("code", "")))
    in_candidates = answered in candidates
    code = answered if in_candidates else (voted or answered)
    try:
        confidence = max(0.0, min(1.0, float(result.data.get("confidence", 0))))
    except (TypeError, ValueError):
        confidence = 0.0
    return Decision(
        code=code,
        precedent=str(result.data.get("precedent", "none")),
        confidence=confidence,
        rationale=str(result.data.get("rationale", "")),
        voted=voted,
        answered=answered,
        in_candidates=in_candidates,
        fallback=not in_candidates,
        error=None,
        prompt_tokens=result.prompt_tokens,
        output_tokens=result.output_tokens,
        seconds=result.seconds,
    )
