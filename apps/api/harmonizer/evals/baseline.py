"""Retrieval-only baselines. No LLM is involved, so these are the floor every
later stage has to beat.

The prediction is the code of the top retrieved candidate. Top-3 counts a hit
when any of the first three distinct codes matches. Every retrieval mode runs
over both candidate pools: past rulings (precedents) and ten-digit tariff lines.
"""

import json
from datetime import UTC, datetime

import typer
from rich import print
from rich.table import Table

from harmonizer.db import connect
from harmonizer.evals.data import REPORTS, load_items
from harmonizer.evals.recall import _parse_modes
from harmonizer.ollama import EMBED_MODEL
from harmonizer.rerank import Reranker
from harmonizer.retrieval import Retriever

LEVELS = (2, 4, 6, 8, 10)
POOLS = ("lines", "precedents")


def split(dev: int = 100, test: int = 1000, since: str = "2025-01-01") -> None:
    """Deterministic dev/test split of recent, single-code NY classification rulings."""
    with connect() as conn, conn.transaction():
        rows = conn.execute(
            """
            SELECT ruling_number, tariff_codes[1]
            FROM cross_ruling
            WHERE collection = 'ny' AND categories ILIKE '%%classification%%'
              AND cardinality(tariff_codes) = 1 AND length(tariff_codes[1]) = 10
              AND ruling_date >= %s AND embedding IS NOT NULL
            ORDER BY md5(ruling_number)
            LIMIT %s
            """,
            (since, dev + test),
        ).fetchall()
        if len(rows) < dev + test:
            print(f"[red]only {len(rows)} eligible rulings, need {dev + test}[/red]")
            raise typer.Exit(1)
        conn.execute("DELETE FROM eval_split")
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO eval_split (ruling_number, split, gold_code) VALUES (%s, %s, %s)",
                [(n, "dev" if i < dev else "test", code) for i, (n, code) in enumerate(rows)],
            )
    print(f"split written: {dev} dev, {test} test, rulings dated >= {since}")


def _dedupe(codes: list[str]) -> list[str]:
    seen: list[str] = []
    for c in codes:
        if c not in seen:
            seen.append(c)
    return seen


def _score(pairs: list[tuple[str, list[str]]]) -> dict[str, float]:
    n = len(pairs)
    out: dict[str, float] = {}
    for level in LEVELS:
        out[f"top1@{level}"] = sum(1 for g, p in pairs if p and p[0][:level] == g[:level]) / n
        hit3 = sum(1 for g, p in pairs if any(x[:level] == g[:level] for x in p[:3]))
        out[f"top3@{level}"] = hit3 / n
    return out


def baseline(
    split_name: str = typer.Option("test", "--split"),
    k: int = 5,
    modes: str = typer.Option("dense,bm25,hybrid", help="Subset of dense,bm25,hybrid,rerank"),
) -> None:
    """Score the retrieval-only baselines on a split and write a JSON report."""
    with connect() as conn:
        conn.execute("SET hnsw.ef_search = 200")
        chosen = _parse_modes(modes)
        retriever = Retriever(conn, Reranker() if "rerank" in chosen else None)
        items = load_items(conn, split_name)
        methods: dict[str, dict[str, float]] = {}
        for pool in POOLS:
            for mode in chosen:
                pairs: list[tuple[str, list[str]]] = []
                for item in items:
                    if pool == "precedents":
                        cands = retriever.precedents(
                            item.text,
                            item.emb,
                            k,
                            mode,
                            before=item.ruling_date,
                            exclude=item.ruling_number,
                        )
                    else:
                        cands = retriever.lines(item.text, item.emb, k, mode)
                    pairs.append((item.gold, _dedupe([c.code for c in cands])))
                methods[f"{pool}-{mode}"] = _score(pairs)

    table = Table(title=f"retrieval-only baselines, split={split_name}, n={len(items)}")
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

    REPORTS.mkdir(parents=True, exist_ok=True)
    today = datetime.now(tz=UTC).date().isoformat()
    out = REPORTS / f"baseline-{split_name}-{today}.json"
    out.write_text(
        json.dumps(
            {
                "date": today,
                "split": split_name,
                "n": len(items),
                "k": k,
                "embed_model": EMBED_MODEL,
                "methods": methods,
            },
            indent=2,
        )
    )
    print(f"report written to {out.relative_to(REPORTS.parents[1])}")
