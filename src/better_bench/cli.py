from __future__ import annotations

import json
from dataclasses import asdict
from datetime import date
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .diagnostics import benchmark_pair_residuals, pairwise_benchmark_diagnostics, taxonomy_fit
from .factors import fit_missing_pca
from .io import load_benchmarks, load_models, load_observations
from .novelty import (
    comparable_benchmark_residuals,
    summarize_global_novelty,
    summarize_model_novelty,
)
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
    results = score_models(load_models(models), load_benchmarks(benchmarks), load_observations(observations), as_of=as_of_date)
    table = Table(title=f"Better Bench — {as_of_date.isoformat()}")
    table.add_column("Model")
    table.add_column("Score", justify="right")
    table.add_column("95% interval", justify="right")
    table.add_column("Coverage", justify="right")
    table.add_column("Effective benches", justify="right")
    for item in results:
        score_text = "—" if item.general_score is None else f"{item.general_score:.2f}"
        interval = "—" if item.ci_low is None else f"{item.ci_low:.2f}–{item.ci_high:.2f}"
        table.add_row(item.model_name, score_text, interval, f"{100 * item.coverage:.1f}%", f"{item.effective_benchmarks:.2f}")
    console.print(table)
    if output_json:
        output_json.write_text(json.dumps([item.model_dump(mode="json") for item in results], indent=2), encoding="utf-8")
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
    rows = pairwise_benchmark_diagnostics(benchmark_defs, observation_rows, minimum_overlap=minimum_overlap)
    fit, pair_count = taxonomy_fit(benchmark_defs, observation_rows, minimum_overlap=minimum_overlap)
    fit_text = "insufficient data" if fit is None else f"{fit:+.3f}"
    console.print(f"Taxonomy fit (Spearman of loading similarity vs |observed rank correlation|): [bold]{fit_text}[/bold] across {pair_count} comparable benchmark pairs.")
    console.print("Treat this as a structural diagnostic, not a validation p-value; small overlaps and benchmark saturation can dominate it.")
    table = Table(title=f"Benchmark pair diagnostics (minimum overlap={minimum_overlap})")
    table.add_column("Left")
    table.add_column("Right")
    table.add_column("n", justify="right")
    table.add_column("Pearson", justify="right")
    table.add_column("Spearman", justify="right")
    table.add_column("Taxonomy sim.", justify="right")
    for row in rows[:top]:
        table.add_row(row.left, row.right, str(row.overlap), f"{row.pearson:+.3f}", f"{row.spearman:+.3f}", f"{row.loading_similarity:.3f}")
    console.print(table)
    if output_csv:
        import pandas as pd

        pd.DataFrame([row.__dict__ for row in rows]).to_csv(output_csv, index=False)
        console.print(f"Wrote {output_csv}")
    if (left is None) != (right is None):
        raise typer.BadParameter("--left and --right must be supplied together")
    if left and right:
        residuals = benchmark_pair_residuals(observation_rows, left, right, minimum_overlap=minimum_overlap)
        residual_table = Table(title=f"Residuals: {right} predicted from {left}")
        residual_table.add_column("Model")
        residual_table.add_column(left, justify="right")
        residual_table.add_column(right, justify="right")
        residual_table.add_column("Predicted", justify="right")
        residual_table.add_column("Residual", justify="right")
        residual_table.add_column("z", justify="right")
        for row in residuals:
            residual_table.add_row(row.model_id, f"{row.left_score:.2f}", f"{row.right_score:.2f}", f"{row.predicted_right:.2f}", f"{row.residual:+.2f}", f"{row.standardized_residual:+.2f}")
        console.print(residual_table)


@app.command()
def factor(
    benchmarks: Path = typer.Option(..., exists=True, readable=True),
    observations: Path = typer.Option(..., exists=True, readable=True),
    rank: int = typer.Option(3, min=1, help="Number of latent factors to fit."),
    minimum_models: int = typer.Option(5, min=3, help="Models required per retained benchmark."),
    minimum_benchmarks: int = typer.Option(5, min=2, help="Benchmarks required per retained model."),
    top: int = typer.Option(20, min=1, help="Number of factor-1 models/loadings to print."),
) -> None:
    """Fit a sparse, standardized low-rank factor diagnostic."""
    result = fit_missing_pca(
        load_benchmarks(benchmarks),
        load_observations(observations),
        rank=rank,
        minimum_models_per_benchmark=minimum_models,
        minimum_benchmarks_per_model=minimum_benchmarks,
    )
    console.print(
        f"Retained matrix: [bold]{len(result.models)} models × {len(result.benchmarks)} benchmarks[/bold] | "
        f"{result.observed_cells}/{result.possible_cells} observed ({100 * result.density:.1f}%)."
    )
    for index, value in enumerate(result.explained_variance, start=1):
        console.print(f"Cumulative observed variance explained by rank {index}: [bold]{100 * value:.1f}%[/bold]")

    model_table = Table(title="Factor 1 — model scores")
    model_table.add_column("Model")
    model_table.add_column("Score", justify="right")
    factor_one = result.model_scores["factor_1"].sort_values(ascending=False)
    for model_id, value in factor_one.head(top).items():
        model_table.add_row(str(model_id), f"{float(value):+.3f}")
    console.print(model_table)

    loading_table = Table(title="Factor 1 — benchmark loadings")
    loading_table.add_column("Benchmark")
    loading_table.add_column("Loading", justify="right")
    loading_one = result.benchmark_loadings["factor_1"]
    order = loading_one.abs().sort_values(ascending=False).head(top).index
    for benchmark_id in order:
        loading_table.add_row(str(benchmark_id), f"{float(loading_one.loc[benchmark_id]):+.3f}")
    console.print(loading_table)


@app.command()
def novelty(
    benchmarks: Path = typer.Option(..., exists=True, readable=True),
    models: Path = typer.Option(..., exists=True, readable=True),
    observations: Path = typer.Option(..., exists=True, readable=True),
    minimum_overlap: int = typer.Option(5, min=3, help="Other shared models needed to calibrate a benchmark pair."),
    minimum_similarity: float = typer.Option(0.20, min=0.0, max=1.0, help="Minimum capability-loading cosine similarity."),
    minimum_correlation: float = typer.Option(0.20, min=-1.0, max=1.0, help="Minimum positive empirical comparator correlation."),
    likely_unseen_days: int = typer.Option(45, min=0, help="Release lead window that counts as suggestive, not proven, non-exposure."),
    top: int = typer.Option(30, min=1, help="Number of low-exposure residuals to show."),
    output_json: Path | None = typer.Option(None, help="Optional machine-readable report."),
) -> None:
    """Test whether low-exposure/new benchmarks produce unexpected model underperformance."""
    model_defs = load_models(models)
    benchmark_defs = load_benchmarks(benchmarks)
    observation_rows = load_observations(observations)
    rows = comparable_benchmark_residuals(
        model_defs,
        benchmark_defs,
        observation_rows,
        minimum_overlap=minimum_overlap,
        minimum_similarity=minimum_similarity,
        minimum_correlation=minimum_correlation,
        likely_unseen_days=likely_unseen_days,
    )
    global_summary = summarize_global_novelty(rows)
    summaries = summarize_model_novelty(rows)
    console.print("Novelty residuals are target scores minus cross-predictions from capability-similar benchmarks. Negative means underperformance relative to comparable evidence.")
    console.print(
        f"Usable residuals: [bold]{len(rows)}[/bold] | strong unseen={global_summary.strong_unseen_count}, "
        f"protected={global_summary.protected_count}, suggestive={global_summary.suggestive_unseen_count}, "
        f"possible exposure={global_summary.possible_exposure_count}."
    )
    gap = "—" if global_summary.novelty_gap is None else f"{global_summary.novelty_gap:+.2f}"
    interval = "—" if global_summary.ci_low is None else f"{global_summary.ci_low:+.2f} to {global_summary.ci_high:+.2f}"
    console.print(f"Descriptive global novelty gap: [bold]{gap}[/bold] (naive 95% interval {interval}). Do not treat this as a p-value; residuals are clustered by model and benchmark.")
    table = Table(title="Model novelty robustness")
    table.add_column("Model")
    table.add_column("Strong", justify="right")
    table.add_column("Protected", justify="right")
    table.add_column("Suggestive", justify="right")
    table.add_column("Exposed?", justify="right")
    table.add_column("Novel residual", justify="right")
    table.add_column("Exposure residual", justify="right")
    table.add_column("Gap", justify="right")
    for row in summaries:
        table.add_row(
            row.model_id,
            str(row.strong_unseen_count),
            str(row.protected_count),
            str(row.suggestive_unseen_count),
            str(row.possible_exposure_count),
            "—" if row.broad_novel_mean_residual is None else f"{row.broad_novel_mean_residual:+.2f}",
            "—" if row.possible_exposure_mean_residual is None else f"{row.possible_exposure_mean_residual:+.2f}",
            "—" if row.novelty_gap is None else f"{row.novelty_gap:+.2f}",
        )
    console.print(table)
    low_exposure = [row for row in rows if row.exposure_group in {"strong_unseen", "protected", "suggestive_unseen"}]
    low_exposure.sort(key=lambda row: row.residual)
    residual_table = Table(title="Most negative low-exposure residuals")
    residual_table.add_column("Model")
    residual_table.add_column("Benchmark")
    residual_table.add_column("Tier")
    residual_table.add_column("Observed", justify="right")
    residual_table.add_column("Predicted", justify="right")
    residual_table.add_column("Residual", justify="right")
    residual_table.add_column("Comparators", justify="right")
    for row in low_exposure[:top]:
        residual_table.add_row(row.model_id, row.benchmark_id, row.exposure_tier.value, f"{row.observed_score:.2f}", f"{row.predicted_score:.2f}", f"{row.residual:+.2f}", str(row.comparator_count))
    console.print(residual_table)
    if output_json:
        payload = {
            "global": asdict(global_summary),
            "models": [asdict(row) for row in summaries],
            "residuals": [asdict(row) for row in rows],
        }
        output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        console.print(f"Wrote {output_json}")


if __name__ == "__main__":
    app()
