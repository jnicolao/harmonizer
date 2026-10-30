"""Does aggregating over neighbours beat trusting the single nearest ruling?

Fused precedent retrieval is fixed; only the step from candidates to a
prediction changes. k is swept for both voting schemes.
"""

import json
from datetime import UTC, datetime

import typer
from rich import print
from rich.table import Table

from harmonizer.db import connect
from harmonizer.evals.baseline import LEVELS, _score
from harmonizer.evals.data import REPORTS, load_items
from harmonizer.retrieval import Retriever
from harmonizer.vote import hierarchical, nearest, vote

KS = (3, 5, 10, 20)


def vote_eval(split_name: str = typer.Option("test", "--split")) -> None:
    """Score nearest, flat vote and hierarchical vote over fused precedents."""
    with connect() as conn:
        conn.execute("SET hnsw.ef_search = 400")
        retriever = Retriever(conn)
        items = load_items(conn, split_name)
        lists = [
            (
                it.gold,
                retriever.precedents(
                    it.text,
                    it.emb,
                    max(KS),
                    "hybrid",
                    before=it.ruling_date,
                    exclude=it.ruling_number,
                ),
            )
            for it in items
        ]

    methods = {"nearest": _score([(g, nearest(c)) for g, c in lists])}
    for k in KS:
        methods[f"vote@{k}"] = _score([(g, vote(c, k)) for g, c in lists])
    for k in KS:
        methods[f"hierarchical@{k}"] = _score([(g, hierarchical(c, k)) for g, c in lists])

    table = Table(title=f"precedent aggregation, split={split_name}, n={len(items)}, fused hybrid")
    table.add_column("method")
    for level in LEVELS:
        table.add_column(f"top1@{level}", justify="right")
    table.add_column("top3@6", justify="right")
    table.add_column("top3@10", justify="right")
    for name, m in methods.items():
        table.add_row(
            name,
            *(f"{m[f'top1@{level}']:.1%}" for level in LEVELS),
            f"{m['top3@6']:.1%}",
            f"{m['top3@10']:.1%}",
        )
    print(table)

    today = datetime.now(tz=UTC).date().isoformat()
    out = REPORTS / f"vote-{split_name}-{today}.json"
    out.write_text(
        json.dumps(
            {
                "date": today,
                "split": split_name,
                "n": len(items),
                "retrieval": "precedents-hybrid",
                "ks": list(KS),
                "methods": methods,
            },
            indent=2,
        )
    )
    print(f"report written to {out.relative_to(REPORTS.parents[1])}")
