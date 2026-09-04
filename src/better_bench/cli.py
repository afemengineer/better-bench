from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .io import load_benchmarks, load_models, load_observations
from .scoring import score_models

app = typer.Typer(no_args_is_help=True, help="Capability-oriented AI benchmark analysis.")
console = Console()


@app.callback()
def main() -> None:
    """Better Bench command-line interface."""


@app.command()
def score(
    benchmarks: Path = typer.Option(..., exists=True, readable=True),
    models: Path = typer.Option(..., exists=True, readable=True),
    observations: Path = typer.Option(..., exists=True, readable=True),
    as_of: str | None = typer.Option(None, help="Evaluation date in YYYY-MM-DD format."),
    output_json: Path | None = typer.Option(None, help="Optional path for machine-readable output."),
) -> None:
    """Score models from benchmark definitions and observations."""
    as_of_date = date.fromisoformat(as_of) if as_of else date.today()
    results = score_models(
        load_models(models),
        load_benchmarks(benchmarks),
        load_observations(observations),
        as_of=as_of_date,
    )
    table = Table(title=f"Better Bench — {as_of_date.isoformat()}")
    table.add_column("Model")
    table.add_column("Score", justify="right")
    table.add_column("95% interval", justify="right")
    table.add_column("Coverage", justify="right")
    table.add_column("Effective benches", justify="right")
    for item in results:
        score_text = "—" if item.general_score is None else f"{item.general_score:.2f}"
        interval = "—" if item.ci_low is None else f"{item.ci_low:.2f}–{item.ci_high:.2f}"
        table.add_row(
            item.model_name,
            score_text,
            interval,
            f"{100 * item.coverage:.1f}%",
            f"{item.effective_benchmarks:.2f}",
        )
    console.print(table)
    if output_json:
        output_json.write_text(
            json.dumps([item.model_dump(mode="json") for item in results], indent=2),
            encoding="utf-8",
        )
        console.print(f"Wrote {output_json}")


if __name__ == "__main__":
    app()
