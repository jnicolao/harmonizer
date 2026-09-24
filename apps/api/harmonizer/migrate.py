from pathlib import Path

import typer
from rich import print

from harmonizer.db import connect

app = typer.Typer(help="Database maintenance")
SQL_DIR = Path(__file__).resolve().parent.parent / "sql"


@app.command()
def migrate() -> None:
    """Apply every sql/*.sql in order. The files are idempotent, so re-running is safe."""
    with connect() as conn:
        for path in sorted(SQL_DIR.glob("*.sql")):
            conn.execute(path.read_text())
            print(f"applied {path.name}")
