from dataclasses import dataclass
from datetime import date
from pathlib import Path


@dataclass(frozen=True)
class EvalItem:
    ruling_number: str
    gold: str
    ruling_date: date
    text: str
    emb: str


def load_items(conn, split: str) -> list[EvalItem]:
    rows = conn.execute(
        """
        SELECT e.ruling_number, e.gold_code, r.ruling_date, r.query_text, r.embedding::text
        FROM eval_split e JOIN cross_ruling r USING (ruling_number)
        WHERE e.split = %s
        ORDER BY e.ruling_number
        """,
        (split,),
    ).fetchall()
    return [EvalItem(*row) for row in rows]


def hit(codes: list[str], gold: str, level: int) -> bool:
    return any(c[:level] == gold[:level] for c in codes)


REPORTS = Path(__file__).resolve().parents[4] / "evals" / "reports"
