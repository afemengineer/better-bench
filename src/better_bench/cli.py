from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .diagnostics import benchmark_pair_residuals, pairwise_benchmark_diagnostics, taxonomy_fit
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


@app.command()
def diagnose(
    benchmarks: Path = typer.Option(..., exists=True, readable=True),
    observations: Path = typer.Option(..., exists=True, readable=True),
    minimum_overlap: int = typer.Option(4, min=3, help="Minimum shared models per benchmark pair."),
    top: int = typer.Option(20, min=1, help="Number of benchmark pairs to print."),
    left: str | None = typer.Option(None, help="Optional benchmark id for residual analysis."),
    right: str | None = typer.Option(None, help="Optional benchmark id for residual analysis."),
    output_csv: Path | None = typer.Option(None, help="Optional pairwise diagnostics CSV."),
) -> None:
    """Inspect empirical benchmark relationships and taxonomy agreement."""
    benchmark_defs = load_benchmarks(benchmarks)
    observation_rows = load_observations(observations)
    rows = pairwise_benchmark_diagnostics(
        benchmark_defs,
        observation_rows,
        minimum_overlap=minimum_overlap,
    )
    fit, pair_count = taxonomy_fit(
        benchmark_defs,
        observation_rows,
        minimum_overlap=minimum_overlap,
    )

    fit_text = "insufficient data" if fit is None else f"{fit:+.3f}"
    console.print(
        f"Taxonomy fit (Spearman of loading similarity vs |observed rank correlation|): "
        f"[bold]{fit_text}[/bold] across {pair_count} comparable benchmark pairs."
    )
    console.print(
        "Treat this as a structural diagnostic, not a validation p-value; small overlaps and "
        "benchmark saturation can dominate it."
    )

    table = Table(title=f"Benchmark pair diagnostics (minimum overlap={minimum_overlap})")
    table.add_column("Left")
    table.add_column("Right")
    table.add_column("n", justify="right")
    table.add_column("Pearson", justify="right")
    table.add_column("Spearman", justify="right")
    table.add_column("Taxonomy sim.", justify="right")
    for row in rows[:top]:
        table.add_row(
            row.left,
            row.right,
            str(row.overlap),
            f"{row.pearson:+.3f}",
            f"{row.spearman:+.3f}",
            f"{row.loading_similarity:.3f}",
        )
    console.print(table)

    if output_csv:
        import pandas as pd

        pd.DataFrame([row.__dict__ for row in rows]).to_csv(output_csv, index=False)
        console.print(f"Wrote {output_csv}")

    if (left is None) != (right is None):
        raise typer.BadParameter("--left and --right must be supplied together")
    if left and right:
        residuals = benchmark_pair_residuals(
            observation_rows,
            left,
            right,
            minimum_overlap=minimum_overlap,
        )
        residual_table = Table(title=f"Residuals: {right} predicted from {left}")
        residual_table.add_column("Model")
        residual_table.add_column(left, justify="right")
        residual_table.add_column(right, justify="right")
        residual_table.add_column("Predicted", justify="right")
        residual_table.add_column("Residual", justify="right")
        residual_table.add_column("z", justify="right")
        for row in residuals:
            residual_table.add_row(
                row.model_id,
                f"{row.left_score:.2f}",
                f"{row.right_score:.2f}",
                f"{row.predicted_right:.2f}",
                f"{row.residual:+.2f}",
                f"{row.standardized_residual:+.2f}",
            )
        console.print(residual_table)


if __name__ == "__main__":
    app()
