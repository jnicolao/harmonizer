import typer

from harmonizer.evals import baseline, recall

app = typer.Typer(help="Evaluation harness")
app.command()(baseline.split)
app.command()(baseline.baseline)
app.command()(recall.recall)
