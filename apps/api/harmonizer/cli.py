import typer

from harmonizer import classify, index, migrate
from harmonizer.evals import cli as evals_cli
from harmonizer.ingest import cross, hts

app = typer.Typer(no_args_is_help=True, help="Harmonizer command line")
app.command("classify")(classify.classify)
app.add_typer(migrate.app, name="db")
app.add_typer(hts.app, name="hts")
app.add_typer(cross.app, name="cross")
app.add_typer(index.app, name="index")
app.add_typer(evals_cli.app, name="eval")
