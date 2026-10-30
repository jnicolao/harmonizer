"""Does a cross-encoder reranker improve on fused retrieval?

Five configurations over the fused top-50, on both candidate pools, varying
what text each side of the pair sees. Fused order is the control. The report
records recall, top-1 and seconds per query for each.
"""

import gc
import json
import time
from datetime import UTC, datetime

import numpy as np
import typer
from rich import print
from rich.table import Table

from harmonizer.db import connect
from harmonizer.evals.data import REPORTS, hit, load_items
from harmonizer.rerank import Reranker
from harmonizer.retrieval import Candidate, Retriever

DEPTH = 50


def _subject_only(text: str) -> str:
    return text.split(". ")[0]


def _metrics(rows: list[tuple[str, list[str]]]) -> dict[str, float]:
    n = len(rows)
    return {
        "recall@5@6": sum(hit(codes[:5], g, 6) for g, codes in rows) / n,
        "recall@10@6": sum(hit(codes[:10], g, 6) for g, codes in rows) / n,
        "top1@6": sum(bool(codes) and codes[0][:6] == g[:6] for g, codes in rows) / n,
        "top1@10": sum(bool(codes) and codes[0] == g for g, codes in rows) / n,
    }


def rerank_ablation(split_name: str = typer.Option("dev", "--split")) -> None:
    """Compare fused order against five cross-encoder reranking configurations."""
    with connect() as conn:
        conn.execute("SET hnsw.ef_search = 400")
        retriever = Retriever(conn)
        subject = dict(conn.execute("SELECT ruling_number, subject FROM cross_ruling").fetchall())
        items = load_items(conn, split_name)
        pools: dict[str, list[tuple[str, str, list[Candidate]]]] = {
            "precedents": [
                (
                    it.text,
                    it.gold,
                    retriever.precedents(
                        it.text,
                        it.emb,
                        DEPTH,
                        "hybrid",
                        before=it.ruling_date,
                        exclude=it.ruling_number,
                    ),
                )
                for it in items
            ],
            "lines": [
                (it.text, it.gold, retriever.lines(it.text, it.emb, DEPTH, "hybrid"))
                for it in items
            ],
        }

    results: dict[str, dict[str, float]] = {}
    for pool, data in pools.items():
        results[f"{pool} fused (control)"] = _metrics(
            [(g, [c.code for c in cands]) for _, g, cands in data]
        ) | {"seconds_per_query": 0.0}

    def cand_subject(c: Candidate) -> str:
        return _subject_only(subject.get(c.key, c.text))

    configs = [
        ("precedents", "desc vs desc", lambda t: t, lambda c: c.text, 512),
        ("precedents", "subject vs subject", _subject_only, cand_subject, 64),
        ("precedents", "first 320 chars both", lambda t: t[:320], lambda c: c.text[:320], 192),
        ("lines", "desc vs path", lambda t: t, lambda c: c.text, 512),
        ("lines", "first 320 chars vs path", lambda t: t[:320], lambda c: c.text, 256),
    ]
    device = None
    for pool, name, query_fn, cand_fn, max_length in configs:
        reranker = Reranker(max_length=max_length)
        device = reranker.device
        started = time.perf_counter()
        rows: list[tuple[str, list[str]]] = []
        for text, gold, cands in pools[pool]:
            order = np.argsort(-reranker.score([(query_fn(text), cand_fn(c)) for c in cands]))
            rows.append((gold, [cands[i].code for i in order]))
        per_query = (time.perf_counter() - started) / len(items)
        results[f"{pool} rerank {name}"] = _metrics(rows) | {"seconds_per_query": per_query}
        print(f"scored {pool} rerank {name} at {per_query:.2f}s per query")
        del reranker
        gc.collect()

    table = Table(title=f"reranker ablation, split={split_name}, n={len(items)}, fused top-{DEPTH}")
    for col in ("configuration", "recall@5@6", "recall@10@6", "top1@6", "top1@10", "s/query"):
        table.add_column(col, justify="right" if col != "configuration" else "left")
    for name, m in results.items():
        table.add_row(
            name,
            f"{m['recall@5@6']:.1%}",
            f"{m['recall@10@6']:.1%}",
            f"{m['top1@6']:.1%}",
            f"{m['top1@10']:.1%}",
            f"{m['seconds_per_query']:.2f}",
        )
    print(table)

    today = datetime.now(tz=UTC).date().isoformat()
    out = REPORTS / f"rerank-ablation-{split_name}-{today}.json"
    out.write_text(
        json.dumps(
            {
                "date": today,
                "split": split_name,
                "n": len(items),
                "depth": DEPTH,
                "reranker": {
                    "model": "BAAI/bge-reranker-v2-m3",
                    "device": device,
                    "dtype": "float16 on mps",
                },
                "results": results,
            },
            indent=2,
        )
    )
    print(f"report written to {out.relative_to(REPORTS.parents[1])}")
