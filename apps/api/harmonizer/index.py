"""Build bge-m3 embeddings for tariff lines and rulings."""

import typer
from rich import print

from harmonizer.db import connect
from harmonizer.ollama import embed, to_pgvector
from harmonizer.rulings import describe

app = typer.Typer(help="Build embeddings with the local Ollama server")
BATCH = 128


def _embed_rows(conn, rows: list[tuple], update_sql: str, label: str) -> None:
    """Embed in batches. The HNSW index is dropped first and rebuilt at the end,
    because maintaining it row by row is far slower than one bulk build."""
    print(f"{label}: {len(rows)} rows to embed")
    conn.execute(f"DROP INDEX IF EXISTS {label}_embedding_idx")
    for start in range(0, len(rows), BATCH):
        chunk = rows[start : start + BATCH]
        vectors = embed([text for _, text in chunk])
        with conn.cursor() as cur:
            params = [(to_pgvector(v), key) for (key, _), v in zip(chunk, vectors, strict=True)]
            cur.executemany(update_sql, params)
        if (start // BATCH) % 25 == 0:
            print(f"  {min(start + BATCH, len(rows))}/{len(rows)}")
    conn.execute("SET maintenance_work_mem = '1GB'")
    conn.execute(
        f"CREATE INDEX IF NOT EXISTS {label}_embedding_idx "
        f"ON {label} USING hnsw (embedding vector_cosine_ops)"
    )
    print(f"{label}: hnsw index built")


@app.command()
def hts() -> None:
    """Embed every numbered tariff line's full path. Skips lines already embedded."""
    with connect() as conn:
        rows = conn.execute(
            "SELECT id, full_path FROM hts_line WHERE htsno <> '' AND embedding IS NULL ORDER BY id"
        ).fetchall()
        update = "UPDATE hts_line SET embedding = %s::vector WHERE id = %s"
        _embed_rows(conn, rows, update, "hts_line")


@app.command()
def rulings() -> None:
    """Derive query_text for each ruling, then embed it. Skips work already done."""
    with connect() as conn:
        todo = conn.execute(
            "SELECT ruling_number, subject, text FROM cross_ruling "
            "WHERE query_text IS NULL AND text IS NOT NULL AND text <> ''"
        ).fetchall()
        with conn.cursor() as cur:
            cur.executemany(
                "UPDATE cross_ruling SET query_text = %s WHERE ruling_number = %s",
                [(describe(subject, text), number) for number, subject, text in todo],
            )
        print(f"query_text set for {len(todo)} rulings")

        rows = conn.execute(
            "SELECT ruling_number, query_text FROM cross_ruling "
            "WHERE embedding IS NULL AND query_text IS NOT NULL AND query_text <> '' "
            "ORDER BY ruling_date DESC"
        ).fetchall()
        _embed_rows(
            conn,
            rows,
            "UPDATE cross_ruling SET embedding = %s::vector WHERE ruling_number = %s",
            "cross_ruling",
        )
