from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .schema import BenchmarkDefinition, BenchmarkObservation
from .scoring import normalize_score


@dataclass(frozen=True)
class FactorModelResult:
    models: list[str]
    benchmarks: list[str]
    observed_cells: int
    possible_cells: int
    density: float
    explained_variance: list[float]
    model_scores: pd.DataFrame
    benchmark_loadings: pd.DataFrame


def _normalized_matrix(
    benchmarks: list[BenchmarkDefinition],
    observations: list[BenchmarkObservation],
) -> pd.DataFrame:
    benchmark_by_id = {row.id: row for row in benchmarks}
    rows: list[tuple[str, str, float]] = []
    for observation in observations:
        benchmark = benchmark_by_id.get(observation.benchmark_id)
        if benchmark is None:
            continue
        rows.append(
            (
                observation.model_id,
                observation.benchmark_id,
                normalize_score(observation.score, benchmark),
            )
        )
    if not rows:
        return pd.DataFrame(dtype=float)
    return pd.DataFrame(rows, columns=["model_id", "benchmark_id", "score"]).pivot_table(
        index="model_id",
        columns="benchmark_id",
        values="score",
        aggfunc="mean",
    )


def _filter_matrix(
    frame: pd.DataFrame,
    *,
    minimum_models_per_benchmark: int,
    minimum_benchmarks_per_model: int,
) -> pd.DataFrame:
    current = frame.copy()
    for _ in range(8):
        before = current.shape
        current = current.loc[
            current.notna().sum(axis=1) >= minimum_benchmarks_per_model,
            current.notna().sum(axis=0) >= minimum_models_per_benchmark,
        ]
        if current.shape == before:
            break
    return current


def fit_missing_pca(
    benchmarks: list[BenchmarkDefinition],
    observations: list[BenchmarkObservation],
    *,
    rank: int = 3,
    minimum_models_per_benchmark: int = 5,
    minimum_benchmarks_per_model: int = 5,
    maximum_iterations: int = 100,
    tolerance: float = 1e-7,
) -> FactorModelResult:
    """Fit a diagnostic low-rank factor model to a sparse benchmark matrix.

    Scores are first normalized to each benchmark's fixed 0-100 scale and then
    z-scored *within benchmark*. Missing cells are initialized at the benchmark
    mean (zero after standardization) and iteratively imputed from a rank-k SVD.

    This is deliberately a structural diagnostic, not the final Better Bench
    estimator. Missingness is not random and source/harness effects remain.
    """
    if rank < 1:
        raise ValueError("rank must be at least 1")

    frame = _filter_matrix(
        _normalized_matrix(benchmarks, observations),
        minimum_models_per_benchmark=minimum_models_per_benchmark,
        minimum_benchmarks_per_model=minimum_benchmarks_per_model,
    )
    if frame.empty or min(frame.shape) < 2:
        raise ValueError("Insufficient overlapping data for factor analysis")

    means = frame.mean(axis=0, skipna=True)
    stds = frame.std(axis=0, skipna=True, ddof=0)
    keep = stds[stds > 1e-8].index
    frame = frame.loc[:, keep]
    means = means.loc[keep]
    stds = stds.loc[keep]
    if frame.empty:
        raise ValueError("All retained benchmarks have near-zero variance")

    standardized = (frame - means) / stds
    observed = standardized.notna().to_numpy()
    values = standardized.to_numpy(dtype=float)
    filled = np.where(observed, values, 0.0)

    effective_rank = min(rank, min(filled.shape))
    missing = ~observed
    for _ in range(maximum_iterations):
        u, singular, vt = np.linalg.svd(filled, full_matrices=False)
        reconstruction = (u[:, :effective_rank] * singular[:effective_rank]) @ vt[:effective_rank]
        if not missing.any():
            filled = values.copy()
            break
        previous_missing = filled[missing].copy()
        filled[missing] = reconstruction[missing]
        filled[observed] = values[observed]
        delta = float(np.max(np.abs(filled[missing] - previous_missing)))
        if delta < tolerance:
            break

    u, singular, vt = np.linalg.svd(filled, full_matrices=False)
    scores = u[:, :effective_rank] * singular[:effective_rank]
    loadings = vt[:effective_rank].T

    for index in range(effective_rank):
        if float(loadings[:, index].sum()) < 0:
            loadings[:, index] *= -1
            scores[:, index] *= -1

    total_ss = float(np.square(values[observed]).sum())
    explained: list[float] = []
    for current_rank in range(1, effective_rank + 1):
        reconstruction = (u[:, :current_rank] * singular[:current_rank]) @ vt[:current_rank]
        residual_ss = float(np.square(values[observed] - reconstruction[observed]).sum())
        explained.append(1.0 - residual_ss / total_ss if total_ss > 0 else 0.0)

    factor_names = [f"factor_{index + 1}" for index in range(effective_rank)]
    model_scores = pd.DataFrame(scores, index=frame.index, columns=factor_names)
    benchmark_loadings = pd.DataFrame(loadings, index=frame.columns, columns=factor_names)

    observed_cells = int(observed.sum())
    possible_cells = int(observed.size)
    return FactorModelResult(
        models=list(frame.index),
        benchmarks=list(frame.columns),
        observed_cells=observed_cells,
        possible_cells=possible_cells,
        density=observed_cells / possible_cells,
        explained_variance=explained,
        model_scores=model_scores,
        benchmark_loadings=benchmark_loadings,
    )
