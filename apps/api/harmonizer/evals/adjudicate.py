"""Does the LLM adjudicator beat the voting floor on the same items?

Every item records both the model's code and the hierarchical vote it was
given as a prior, so the comparison is paired. Per-item rows are kept in the
report for error analysis.
"""

import json
from datetime import UTC, datetime

import typer
from rich import print
from rich.table import Table

from harmonizer.adjudicate import adjudicate
from harmonizer.db import connect
from harmonizer.evals.baseline import LEVELS
from harmonizer.evals.data import REPORTS, load_items
from harmonizer.ollama import CHAT_MODEL
from harmonizer.retrieval import Retriever


def _top1(rows: list[dict], key: str) -> dict[str, float]:
    n = len(rows)
    return {
        f"top1@{level}": sum(1 for r in rows if r[key] and r[key][:level] == r["gold"][:level]) / n
        for level in LEVELS
    }


def adjudicate_eval(
    split_name: str = typer.Option("dev", "--split"),
    limit: int = typer.Option(0, help="Stop after this many items (0 = whole split)"),
) -> None:
    """Score the adjudicator against hierarchical voting, item by item."""
    with connect() as conn:
        conn.execute("SET hnsw.ef_search = 400")
        retriever = Retriever(conn)
        items = load_items(conn, split_name)
        if limit:
            items = items[:limit]
        rows: list[dict] = []
        for i, it in enumerate(items, 1):
            d = adjudicate(
                retriever, it.text, it.emb, before=it.ruling_date, exclude=it.ruling_number
            )
            rows.append(
                {
                    "ruling": it.ruling_number,
                    "gold": it.gold,
                    "code": d.code,
                    "voted": d.voted or "",
                    "answered": d.answered or "",
                    "in_candidates": d.in_candidates,
                    "fallback": d.fallback,
                    "error": d.error,
                    "agree": d.code == (d.voted or ""),
                    "confidence": d.confidence,
                    "precedent": d.precedent,
                    "rationale": d.rationale,
                    "prompt_tokens": d.prompt_tokens,
                    "output_tokens": d.output_tokens,
                    "seconds": d.seconds,
                }
            )
            if i % 10 == 0 or i == len(items):
                done = sum(1 for r in rows if r["code"][:6] == r["gold"][:6])
                print(
                    f"{i}/{len(items)} adjudicated, running top1@6 {done / i:.1%}, "
                    f"{sum(r['seconds'] for r in rows) / i:.1f}s per item"
                )

    n = len(rows)
    methods = {"adjudicator": _top1(rows, "code"), "hierarchical@20": _top1(rows, "voted")}
    right = [r["confidence"] for r in rows if r["code"][:6] == r["gold"][:6]]
    wrong = [r["confidence"] for r in rows if r["code"][:6] != r["gold"][:6]]
    rates = {
        "valid_output": sum(1 for r in rows if not r["error"]) / n,
        "in_candidates": sum(1 for r in rows if r["in_candidates"]) / n,
        "fallback_to_vote": sum(1 for r in rows if r["fallback"]) / n,
        "agrees_with_vote": sum(1 for r in rows if r["agree"]) / n,
        "mean_confidence_when_right@6": sum(right) / len(right) if right else 0.0,
        "mean_confidence_when_wrong@6": sum(wrong) / len(wrong) if wrong else 0.0,
        "seconds_per_item": sum(r["seconds"] for r in rows) / n,
        "prompt_tokens_per_item": sum(r["prompt_tokens"] for r in rows) / n,
    }

    table = Table(title=f"adjudicator vs voting floor, split={split_name}, n={n}, {CHAT_MODEL}")
    table.add_column("method")
    for level in LEVELS:
        table.add_column(f"top1@{level}", justify="right")
    for name, m in methods.items():
        table.add_row(name, *(f"{m[f'top1@{level}']:.1%}" for level in LEVELS))
    print(table)
    for k, v in rates.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) and v <= 1 else f"  {k}: {v:.1f}")

    today = datetime.now(tz=UTC).date().isoformat()
    out = REPORTS / f"adjudicate-{split_name}-{today}.json"
    out.write_text(
        json.dumps(
            {
                "date": today,
                "split": split_name,
                "n": n,
                "model": CHAT_MODEL,
                "methods": methods,
                "rates": rates,
                "items": rows,
            },
            indent=2,
        )
    )
    print(f"report written to {out.relative_to(REPORTS.parents[1])}")
