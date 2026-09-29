import typer

from harmonizer.evals import baseline, recall, rerank_ablation

app = typer.Typer(help="Evaluation harness")
app.command()(baseline.split)
app.command()(baseline.baseline)
app.command()(recall.recall)
app.command("rerank-ablation")(rerank_ablation.rerank_ablation)
