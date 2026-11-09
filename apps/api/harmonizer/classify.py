"""Classify one product description end to end: embed, retrieve, vote, adjudicate."""

from __future__ import annotations

import typer
from rich import print
from rich.markup import escape

from harmonizer.adjudicate import adjudicate, dotted
from harmonizer.db import connect
from harmonizer.ollama import embed, to_pgvector
from harmonizer.retrieval import Retriever


def _line(conn, code: str) -> tuple[str, str] | None:
    """Deepest tariff line that is a prefix of `code`, with the general duty it inherits."""
    row = conn.execute(
        "SELECT full_path FROM hts_line "
        "WHERE htsno <> '' AND %s LIKE replace(htsno, '.', '') || '%%' "
        "ORDER BY digits DESC LIMIT 1",
        (code,),
    ).fetchone()
    if row is None:
        return None
    duty = conn.execute(
        "SELECT general FROM hts_line "
        "WHERE htsno <> '' AND general <> '' AND %s LIKE replace(htsno, '.', '') || '%%' "
        "ORDER BY digits DESC LIMIT 1",
        (code,),
    ).fetchone()
    return row[0], duty[0] if duty else "not stated"


def classify(
    description: str = typer.Argument(
        ..., help="What the product is, what it is made of and what it is used for"
    ),
) -> None:
    """Classify one product description with the whole pipeline."""
    with connect() as conn:
        conn.execute("SET hnsw.ef_search = 400")
        retriever = Retriever(conn)
        emb = to_pgvector(embed([description])[0])
        d = adjudicate(retriever, description, emb)
        line = _line(conn, d.code)
        ruling = None
        if d.precedent.lower() != "none":
            ruling = conn.execute(
                "SELECT ruling_date, subject FROM cross_ruling WHERE ruling_number = %s",
                (d.precedent,),
            ).fetchone()

    print(f"[bold]{dotted(d.code)}[/bold]  confidence {d.confidence:.2f}")
    if line:
        print(f"  {escape(line[0])}")
        print(f"  general duty: {escape(line[1])}")
    if ruling:
        print(f"  precedent: {d.precedent} ({ruling[0]}) {escape(ruling[1])}")
    elif d.precedent.lower() != "none":
        print(f"  precedent: {escape(d.precedent)}")
    if d.voted and d.voted != d.code:
        print(f"  neighbour vote preferred {dotted(d.voted)}")
    if d.fallback:
        reason = f": {escape(d.error)}" if d.error else ", answer was outside the candidates"
        print(f"  [yellow]fell back to the vote{reason}[/yellow]")
    if d.rationale:
        print(f"  {escape(d.rationale)}")
    print(f"  [dim]{d.seconds:.1f}s, {d.prompt_tokens} prompt tokens[/dim]")
