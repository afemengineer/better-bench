from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .schema import BenchmarkDefinition, BenchmarkObservation


@dataclass(frozen=True)
class BenchmarkPairDiagnostic:
    left: str
    right: str
    overlap: int
    pearson: float
    spearman: float
    loading_similarity: float


@dataclass(frozen=True)
class ResidualDiagnostic:
    model_id: str
    left_score: float
    right_score: float
    predicted_right: float
    residual: float
    standardized_residual: float


def observation_matrix(observations: list[BenchmarkObservation]) -> pd.DataFrame:
    """Return a model × benchmark matrix, averaging repeated protocol-identical rows."""
    if not observations:
        return pd.DataFrame()
    return pd.DataFrame(
        [(o.model_id, o.benchmark_id, o.score) for o in observations],
        columns=["model_id", "benchmark_id", "score"],
    ).pivot_table(
        index="model_id",
        columns="benchmark_id",
        values="score",
        aggfunc="mean",
    )


def _loading_similarity(left: BenchmarkDefinition, right: BenchmarkDefinition) -> float:
    keys = set(left.capability_loadings) | set(right.capability_loadings)
    a = np.asarray([left.capability_loadings.get(key, 0.0) for key in keys], dtype=float)
    b = np.asarray([right.capability_loadings.get(key, 0.0) for key in keys], dtype=float)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def pairwise_benchmark_diagnostics(
    benchmarks: list[BenchmarkDefinition],
    observations: list[BenchmarkObservation],
    *,
    minimum_overlap: int = 4,
) -> list[BenchmarkPairDiagnostic]:
    """Measure observed benchmark similarity against the hand-authored taxonomy."""
    matrix = observation_matrix(observations)
    if matrix.empty:
        return []

    definition_by_id = {benchmark.id: benchmark for benchmark in benchmarks}
    ids = [benchmark_id for benchmark_id in matrix.columns if benchmark_id in definition_by_id]
    rows: list[BenchmarkPairDiagnostic] = []

    for i, left_id in enumerate(ids):
        for right_id in ids[i + 1 :]:
            pair = matrix[[left_id, right_id]].dropna()
            if len(pair) < minimum_overlap:
                continue
            pearson = pair[left_id].corr(pair[right_id], method="pearson")
            spearman = pair[left_id].corr(pair[right_id], method="spearman")
            if pd.isna(pearson) or pd.isna(spearman):
                continue
            rows.append(
                BenchmarkPairDiagnostic(
                    left=left_id,
                    right=right_id,
                    overlap=len(pair),
                    pearson=float(pearson),
                    spearman=float(spearman),
                    loading_similarity=_loading_similarity(
                        definition_by_id[left_id], definition_by_id[right_id]
                    ),
                )
            )
    return sorted(rows, key=lambda row: (-row.overlap, -abs(row.spearman), row.left, row.right))


def taxonomy_fit(
    benchmarks: list[BenchmarkDefinition],
    observations: list[BenchmarkObservation],
    *,
    minimum_overlap: int = 4,
) -> tuple[float | None, int]:
    """
    Return correlation between taxonomy similarity and observed rank correlation.

    This is deliberately only a diagnostic. With few models per benchmark pair it is
    unstable, and high benchmark saturation can make a sensible taxonomy look poor.
    """
    rows = pairwise_benchmark_diagnostics(
        benchmarks,
        observations,
        minimum_overlap=minimum_overlap,
    )
    if len(rows) < 3:
        return None, len(rows)
    expected = pd.Series([row.loading_similarity for row in rows], dtype=float)
    observed = pd.Series([abs(row.spearman) for row in rows], dtype=float)
    fit = expected.corr(observed, method="spearman")
    if pd.isna(fit):
        return None, len(rows)
    return float(fit), len(rows)


def benchmark_pair_residuals(
    observations: list[BenchmarkObservation],
    left_id: str,
    right_id: str,
    *,
    minimum_overlap: int = 4,
) -> list[ResidualDiagnostic]:
    """
    Regress right benchmark on left benchmark and expose model-level disagreements.

    Positive residual = model performs better on `right_id` than its `left_id` score
    would predict within the overlapping cohort.
    """
    matrix = observation_matrix(observations)
    if left_id not in matrix or right_id not in matrix:
        return []
    pair = matrix[[left_id, right_id]].dropna()
    if len(pair) < minimum_overlap:
        return []

    x = pair[left_id].to_numpy(dtype=float)
    y = pair[right_id].to_numpy(dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    predicted = slope * x + intercept
    residuals = y - predicted
    if len(residuals) > 2:
        residual_sd = float(np.std(residuals, ddof=1))
    else:
        residual_sd = 0.0
    if residual_sd <= 1e-12:
        residual_sd = 1.0

    rows = [
        ResidualDiagnostic(
            model_id=model_id,
            left_score=float(left),
            right_score=float(right),
            predicted_right=float(pred),
            residual=float(resid),
            standardized_residual=float(resid / residual_sd),
        )
        for model_id, left, right, pred, resid in zip(
            pair.index, x, y, predicted, residuals, strict=True
        )
    ]
    return sorted(rows, key=lambda row: abs(row.standardized_residual), reverse=True)
