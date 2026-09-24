"""Retrieval-only baselines. No LLM is involved, so these are the floor every
later stage has to beat.

  hts-knn        nearest ten-digit tariff lines to the ruling's product description
  precedent-knn  codes of the nearest earlier rulings (label transfer)
"""

import json
from datetime import date
from pathlib import Path

import typer
from rich import print
from rich.table import Table

from harmonizer.db import connect
from harmonizer.ollama import EMBED_MODEL

app = typer.Typer(help="Evaluation harness")
LEVELS = (2, 4, 6, 8, 10)
REPORTS = Path(__file__).resolve().parents[4] / "evals" / "reports"


@app.command()
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


@app.command()
def baseline(split_name: str = typer.Option("test", "--split"), k: int = 5) -> None:
    """Score the retrieval-only baselines on a split and write a JSON report."""
    with connect() as conn:
        conn.execute("SET hnsw.ef_search = 200")
        items = conn.execute(
            """
            SELECT e.ruling_number, e.gold_code, r.ruling_date, r.embedding::text
            FROM eval_split e JOIN cross_ruling r USING (ruling_number)
            WHERE e.split = %s
            """,
            (split_name,),
        ).fetchall()
        hts_pairs: list[tuple[str, list[str]]] = []
        prec_pairs: list[tuple[str, list[str]]] = []
        for number, gold, ruling_date, emb in items:
            hts = conn.execute(
                "SELECT htsno FROM hts_line WHERE digits = 10 AND embedding IS NOT NULL "
                "ORDER BY embedding <=> %s::vector LIMIT %s",
                (emb, k),
            ).fetchall()
            hts_pairs.append((gold, [h[0].replace(".", "") for h in hts]))
            prec = conn.execute(
                "SELECT tariff_codes[1] FROM cross_ruling "
                "WHERE embedding IS NOT NULL AND cardinality(tariff_codes) = 1 "
                "  AND ruling_number <> %s AND ruling_date < %s "
                "ORDER BY embedding <=> %s::vector LIMIT %s",
                (number, ruling_date, emb, k),
            ).fetchall()
            prec_pairs.append((gold, _dedupe([p[0] for p in prec])))

    methods = {"hts-knn": _score(hts_pairs), "precedent-knn": _score(prec_pairs)}

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
    out = REPORTS / f"baseline-{split_name}-{date.today().isoformat()}.json"
    out.write_text(
        json.dumps(
            {
                "date": date.today().isoformat(),
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
