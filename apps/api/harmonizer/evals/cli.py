import typer

from harmonizer.evals import adjudicate, baseline, recall, rerank_ablation, vote

app = typer.Typer(help="Evaluation harness")
app.command()(baseline.split)
app.command()(baseline.baseline)
app.command()(recall.recall)
app.command("rerank-ablation")(rerank_ablation.rerank_ablation)
app.command("vote")(vote.vote_eval)
app.command("adjudicate")(adjudicate.adjudicate_eval)
