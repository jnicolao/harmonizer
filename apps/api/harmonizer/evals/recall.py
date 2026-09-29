"""Candidate recall: is the correct code among the top-k retrieved candidates?

This is the ceiling for any stage that chooses among candidates, so it is
reported for every retrieval mode and both candidate pools, together with the
time each mode costs per query.
"""

import json
import time
from datetime import UTC, datetime

import typer
from rich import print
from rich.table import Table

from harmonizer.db import connect
from harmonizer.evals.data import REPORTS, hit, load_items
from harmonizer.rerank import Reranker
from harmonizer.retrieval import MODES, Retriever

LEVELS = (4, 6, 10)
POOLS = ("precedents", "lines")


def _parse_modes(modes: str) -> list[str]:
    chosen = [m.strip() for m in modes.split(",") if m.strip()]
    unknown = set(chosen) - set(MODES)
    if unknown:
        raise typer.BadParameter(f"unknown modes {sorted(unknown)}; choose from {MODES}")
    return chosen


def recall(
    split_name: str = typer.Option("test", "--split"),
    ks: str = typer.Option("5,10,20,50", help="Comma-separated k values"),
    modes: str = typer.Option("dense,bm25,hybrid", help="Subset of dense,bm25,hybrid,rerank"),
) -> None:
    """Recall@k at 4, 6 and 10 digits for dense, BM25, hybrid and reranked retrieval."""
    k_values = [int(k) for k in ks.split(",")]
    depth = max(k_values)
    chosen = _parse_modes(modes)
    reranker = Reranker() if "rerank" in chosen else None
    with connect() as conn:
        conn.execute("SET hnsw.ef_search = 400")
        retriever = Retriever(conn, reranker)
        items = load_items(conn, split_name)
        results: dict[str, dict[str, float]] = {}
        for pool in POOLS:
            for mode in chosen:
                hits = {(k, level): 0 for k in k_values for level in LEVELS}
                started = time.perf_counter()
                for item in items:
                    if pool == "precedents":
                        cands = retriever.precedents(
                            item.text,
                            item.emb,
                            depth,
                            mode,
                            before=item.ruling_date,
                            exclude=item.ruling_number,
                        )
                    else:
                        cands = retriever.lines(item.text, item.emb, depth, mode)
                    codes = [c.code for c in cands]
                    for k in k_values:
                        for level in LEVELS:
                            if hit(codes[:k], item.gold, level):
                                hits[(k, level)] += 1
                elapsed = time.perf_counter() - started
                results[f"{pool}-{mode}"] = {
                    f"recall@{k}@{level}": hits[(k, level)] / len(items)
                    for k in k_values
                    for level in LEVELS
                } | {"seconds_per_query": elapsed / len(items)}
                print(f"scored {pool}-{mode} at {elapsed / len(items):.3f}s per query")

    for level in LEVELS:
        table = Table(title=f"recall@k at {level} digits, split={split_name}, n={len(items)}")
        table.add_column("method")
        for k in k_values:
            table.add_column(f"k={k}", justify="right")
        table.add_column("s/query", justify="right")
        for name, m in results.items():
            table.add_row(
                name,
                *(f"{m[f'recall@{k}@{level}']:.1%}" for k in k_values),
                f"{m['seconds_per_query']:.3f}",
            )
        print(table)

    today = datetime.now(tz=UTC).date().isoformat()
    out = REPORTS / f"recall-{split_name}-{today}.json"
    out.write_text(
        json.dumps(
            {
                "date": today,
                "split": split_name,
                "n": len(items),
                "ks": k_values,
                "levels": list(LEVELS),
                "reranker": {"model": reranker.model_name, "device": reranker.device}
                if reranker
                else None,
                "methods": results,
            },
            indent=2,
        )
    )
    print(f"report written to {out.relative_to(REPORTS.parents[1])}")
