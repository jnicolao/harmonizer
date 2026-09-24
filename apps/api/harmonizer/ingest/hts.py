"""Ingest the USITC Harmonized Tariff Schedule into Postgres.

The export lists every schedule row in order with an `indent` level. The
hierarchy is rebuilt from that order so each line gets a readable full path,
which matters because a leaf's own description is often just "Other".
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import httpx
import typer
from psycopg.types.json import Jsonb
from rich import print

from harmonizer.db import connect

EXPORT_URL = "https://hts.usitc.gov/reststop/exportList?from=0101&to=9999&format=JSON&styles=false"

app = typer.Typer(help="USITC HTS ingestion")

INSERT = """
INSERT INTO hts_line (ordinal, htsno, digits, indent, description, full_path, superior, units,
                      general, special, other, footnotes, quota_quantity, additional_duties)
VALUES (%(ordinal)s, %(htsno)s, %(digits)s, %(indent)s, %(description)s, %(full_path)s,
        %(superior)s, %(units)s, %(general)s, %(special)s, %(other)s, %(footnotes)s,
        %(quota_quantity)s, %(additional_duties)s)
"""


def build_rows(raw: list[dict]) -> Iterator[dict]:
    stack: list[tuple[int, str]] = []  # (indent, description) of the current ancestors
    for ordinal, r in enumerate(raw):
        indent = int(r.get("indent") or 0)
        description = (r.get("description") or "").strip()
        while stack and stack[-1][0] >= indent:
            stack.pop()
        stack.append((indent, description))
        htsno = (r.get("htsno") or "").strip()
        yield {
            "ordinal": ordinal,
            "htsno": htsno,
            "digits": len(htsno.replace(".", "")),
            "indent": indent,
            "description": description,
            "full_path": " > ".join(d for _, d in stack),
            "superior": str(r.get("superior")).lower() == "true",
            "units": r.get("units") or [],
            "general": r.get("general") or "",
            "special": r.get("special") or "",
            "other": r.get("other") or "",
            "footnotes": Jsonb(r.get("footnotes") or []),
            "quota_quantity": r.get("quotaQuantity") or "",
            "additional_duties": r.get("additionalDuties") or "",
        }


@app.command()
def ingest(
    file: Path | None = typer.Option(None, help="Local export JSON. Downloaded if omitted."),
) -> None:
    """Load the schedule into hts_line, replacing any previous load."""
    if file is None:
        print(f"downloading {EXPORT_URL}")
        raw = httpx.get(EXPORT_URL, timeout=300).json()
        source = EXPORT_URL
    else:
        raw = json.loads(file.read_text())
        source = str(file)
    rows = list(build_rows(raw))

    with connect() as conn, conn.transaction():
        conn.execute("TRUNCATE hts_line")
        conn.execute(
            "INSERT INTO hts_release (source, row_count) VALUES (%s, %s)", (source, len(rows))
        )
        with conn.cursor() as cur:
            cur.executemany(INSERT, rows)

    leaves = sum(1 for r in rows if r["digits"] == 10)
    print(f"loaded {len(rows)} rows, {leaves} ten-digit lines, from {source}")
