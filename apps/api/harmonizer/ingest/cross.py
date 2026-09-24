"""Ingest CBP CROSS customs rulings.

Two steps, both idempotent:
  index       page the search API for ruling metadata (subject, date, tariff codes)
  fetch-text  download full text for rulings that do not have it yet

The search API caps any single query at 10,000 hits, so the index walks
several collection/term combinations and merges on ruling number.
"""

from __future__ import annotations

import asyncio
import re
from datetime import date

import httpx
import typer
from psycopg.types.json import Jsonb
from rich import print

from harmonizer.db import connect

BASE_URL = "https://rulings.cbp.gov/api"
HEADERS = {"User-Agent": "harmonizer/0.1 (open-source HS classification research)"}
CODE_RE = re.compile(r"\b\d{4}\.\d{2}(?:\.\d{2,4})?\b")

app = typer.Typer(help="CBP CROSS rulings ingestion")

UPSERT = """
INSERT INTO cross_ruling (ruling_number, collection, subject, categories, ruling_date,
                          tariffs, tariff_codes, revoked, raw)
VALUES (%(ruling_number)s, %(collection)s, %(subject)s, %(categories)s, %(ruling_date)s,
        %(tariffs)s, %(tariff_codes)s, %(revoked)s, %(raw)s)
ON CONFLICT (ruling_number) DO UPDATE SET
    subject = EXCLUDED.subject, categories = EXCLUDED.categories,
    ruling_date = EXCLUDED.ruling_date, tariffs = EXCLUDED.tariffs,
    tariff_codes = EXCLUDED.tariff_codes, revoked = EXCLUDED.revoked, raw = EXCLUDED.raw
"""


def parse_codes(tariffs: str) -> list[str]:
    """'8301.10.4000, 8301.10.50' -> ['8301104000', '83011050'] (digits only, order kept)."""
    seen: list[str] = []
    for m in CODE_RE.findall(tariffs or ""):
        code = m.replace(".", "")
        if code not in seen:
            seen.append(code)
    return seen


def to_row(r: dict) -> dict:
    raw_date = r.get("rulingDate") or ""
    return {
        "ruling_number": r["rulingNumber"],
        "collection": (r.get("collection") or "").lower(),
        "subject": r.get("subject") or "",
        "categories": r.get("categories") or "",
        "ruling_date": date.fromisoformat(raw_date[:10]) if raw_date else None,
        "tariffs": r.get("tariffs") or "",
        "tariff_codes": parse_codes(r.get("tariffs") or ""),
        "revoked": bool(r.get("revokedBy")) or bool(r.get("operationallyRevoked")),
        "raw": Jsonb(r),
    }


@app.command()
def index(
    collections: str = typer.Option("NY,HQ", help="Comma-separated CROSS collections"),
    terms: str = typer.Option("classification,tariff", help="Comma-separated search terms"),
    page_size: int = typer.Option(500, help="Search API accepts up to 500"),
    max_pages: int = typer.Option(20, help="20 x 500 reaches the 10,000-hit cap"),
) -> None:
    """Page the search API and upsert ruling metadata."""
    total = 0
    with connect() as conn, httpx.Client(base_url=BASE_URL, headers=HEADERS, timeout=60) as http:
        for collection in collections.split(","):
            for term in terms.split(","):
                for page in range(1, max_pages + 1):
                    resp = http.get(
                        "/search",
                        params={
                            "term": term,
                            "collection": collection,
                            "pageSize": page_size,
                            "page": page,
                            "sortBy": "DATE_DESC",
                        },
                    )
                    resp.raise_for_status()
                    rulings = resp.json().get("rulings") or []
                    if not rulings:
                        break
                    with conn.cursor() as cur:
                        cur.executemany(UPSERT, [to_row(r) for r in rulings])
                    total += len(rulings)
                    print(f"{collection}/{term} page {page}: {len(rulings)} rulings")
        count = conn.execute("SELECT count(*) FROM cross_ruling").fetchone()[0]
    print(f"upserted {total} search rows; {count} distinct rulings in cross_ruling")


async def _fetch_one(http: httpx.AsyncClient, sem: asyncio.Semaphore, number: str) -> str | None:
    async with sem:
        for attempt in range(5):
            try:
                resp = await http.get(f"/ruling/{number}")
                if resp.status_code == 404:
                    return ""  # keep as empty text so it is not retried forever
                resp.raise_for_status()
                return resp.json().get("text") or ""
            except httpx.HTTPError:
                await asyncio.sleep(2**attempt)
        return None


async def _fetch_all(concurrency: int, limit: int, batch: int) -> None:
    with connect() as conn:
        sql = "SELECT ruling_number FROM cross_ruling WHERE text IS NULL ORDER BY ruling_date DESC"
        if limit:
            sql += f" LIMIT {int(limit)}"
        numbers = [row[0] for row in conn.execute(sql).fetchall()]
        print(f"{len(numbers)} rulings need text")

        sem = asyncio.Semaphore(concurrency)
        async with httpx.AsyncClient(base_url=BASE_URL, headers=HEADERS, timeout=60) as http:
            for start in range(0, len(numbers), batch):
                chunk = numbers[start : start + batch]
                texts = await asyncio.gather(*(_fetch_one(http, sem, n) for n in chunk))
                done = [(t, n) for n, t in zip(chunk, texts, strict=True) if t is not None]
                with conn.cursor() as cur:
                    cur.executemany(
                        "UPDATE cross_ruling SET text = %s, fetched_at = now() "
                        "WHERE ruling_number = %s",
                        done,
                    )
                print(f"fetched {start + len(chunk)}/{len(numbers)} ({len(done)} ok)")


@app.command("fetch-text")
def fetch_text(
    concurrency: int = typer.Option(4, help="Parallel requests; keep small to be polite"),
    limit: int = typer.Option(0, help="Stop after this many rulings (0 = all)"),
    batch: int = typer.Option(200, help="Rulings per database write"),
) -> None:
    """Download full text for rulings missing it. Safe to re-run."""
    asyncio.run(_fetch_all(concurrency, limit, batch))
