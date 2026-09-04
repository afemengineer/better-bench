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
    benchmarks: list[BenchmarkDefinition], observations: list[BenchmarkObservation]
) -> pd.DataFrame:
    definitions = {row.id: row for row in benchmarks}
    rows = []
    for observation in observations:
        benchmark = definitions.get(observation.benchmark_id)
        if benchmark is not None:
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
        index="model_id", columns="benchmark_id", values="score", aggfunc="mean"
    )


def _filter_matrix(
    frame: pd.DataFrame,
    minimum_models_per_benchmark: int,
    minimum_benchmarks_per_model: int,
) -> pd.DataFrame:
    current = frame.copy()
    for _ in range(8):
        shape = current.shape
        current = current.loc[
            current.notna().sum(axis=1) >= minimum_benchmarks_per_model,
            current.notna().sum(axis=0) >= minimum_models_per_benchmark,
        ]
        if current.shape == shape:
            break
    return current


def _als(
    values: np.ndarray,
    observed: np.ndarray,
    rank: int,
    ridge: float,
    maximum_iterations: int,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Regularized low-rank fit whose loss contains observed cells only."""
    zero_filled = np.where(observed, values, 0.0)
    left, singular, right_t = np.linalg.svd(zero_filled, full_matrices=False)
    root = np.sqrt(np.maximum(singular[:rank], 1e-12))
    model_factors = left[:, :rank] * root
    benchmark_factors = right_t[:rank].T * root
    identity = np.eye(rank)
    previous = float("inf")

    for _ in range(maximum_iterations):
        for i in range(values.shape[0]):
            selected = observed[i]
            design = benchmark_factors[selected]
            target = values[i, selected]
            model_factors[i] = np.linalg.solve(
                design.T @ design + ridge * identity, design.T @ target
            )
        for j in range(values.shape[1]):
            selected = observed[:, j]
            design = model_factors[selected]
            target = values[selected, j]
            benchmark_factors[j] = np.linalg.solve(
                design.T @ design + ridge * identity, design.T @ target
            )

        prediction = model_factors @ benchmark_factors.T
        error = values[observed] - prediction[observed]
        sse = float(np.square(error).sum())
        if previous < float("inf") and abs(previous - sse) / max(previous, 1e-12) < tolerance:
            break
        previous = sse

    prediction = model_factors @ benchmark_factors.T
    error = values[observed] - prediction[observed]
    return model_factors, benchmark_factors, float(np.square(error).sum())


def fit_missing_pca(
    benchmarks: list[BenchmarkDefinition],
    observations: list[BenchmarkObservation],
    *,
    rank: int = 3,
    minimum_models_per_benchmark: int = 5,
    minimum_benchmarks_per_model: int = 5,
    maximum_iterations: int = 200,
    tolerance: float = 1e-7,
    ridge: float = 0.5,
) -> FactorModelResult:
    """Diagnostic sparse factor model fitted only against observed measurements."""
    if rank < 1:
        raise ValueError("rank must be at least 1")
    if ridge <= 0:
        raise ValueError("ridge must be positive")

    frame = _filter_matrix(
        _normalized_matrix(benchmarks, observations),
        minimum_models_per_benchmark,
        minimum_benchmarks_per_model,
    )
    if frame.empty or min(frame.shape) < 2:
        raise ValueError("Insufficient overlapping data for factor analysis")

    means = frame.mean(axis=0, skipna=True)
    stds = frame.std(axis=0, skipna=True, ddof=0)
    keep = stds[stds > 1e-8].index
    frame = frame.loc[:, keep]
    standardized = (frame - means.loc[keep]) / stds.loc[keep]
    observed = standardized.notna().to_numpy()
    values = standardized.to_numpy(dtype=float)
    effective_rank = min(rank, min(values.shape))
    total_ss = float(np.square(values[observed]).sum())

    explained = []
    factor_one_models = None
    factor_one_benchmarks = None
    for current_rank in range(1, effective_rank + 1):
        model_factors, benchmark_factors, sse = _als(
            values,
            observed,
            current_rank,
            ridge,
            maximum_iterations,
            tolerance,
        )
        explained.append(1.0 - sse / total_ss if total_ss else 0.0)
        if current_rank == 1:
            factor_one_models = model_factors[:, 0].copy()
            factor_one_benchmarks = benchmark_factors[:, 0].copy()

    assert factor_one_models is not None and factor_one_benchmarks is not None
    if float(factor_one_benchmarks.sum()) < 0:
        factor_one_models *= -1
        factor_one_benchmarks *= -1
    scale = float(factor_one_models.std(ddof=0))
    if scale > 1e-12:
        factor_one_models /= scale
        factor_one_benchmarks *= scale
    factor_one_models -= float(factor_one_models.mean())

    observed_cells = int(observed.sum())
    possible_cells = int(observed.size)
    return FactorModelResult(
        models=list(frame.index),
        benchmarks=list(frame.columns),
        observed_cells=observed_cells,
        possible_cells=possible_cells,
        density=observed_cells / possible_cells,
        explained_variance=explained,
        model_scores=pd.DataFrame({"factor_1": factor_one_models}, index=frame.index),
        benchmark_loadings=pd.DataFrame(
            {"factor_1": factor_one_benchmarks}, index=frame.columns
        ),
    )
