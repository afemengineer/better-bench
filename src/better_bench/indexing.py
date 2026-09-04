from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import date

import numpy as np

from .benchmark_quality import BenchmarkTier, rank_benchmarks
from .estimator import EstimatorResult
from .schema import (
    SOURCE_GRADE_WEIGHT,
    BenchmarkAdoptionSnapshot,
    BenchmarkDefinition,
    BenchmarkObservation,
    ModelDefinition,
)
from .scoring import normalize_score


_TIER_WEIGHT = {
    BenchmarkTier.CORE: 1.00,
    BenchmarkTier.EMERGING: 0.90,
    BenchmarkTier.SUPPORTING: 0.55,
    BenchmarkTier.DIAGNOSTIC: 0.20,
}


@dataclass(frozen=True)
class BetterBenchIndexEstimate:
    model_id: str
    index: float
    ci_low: float
    ci_high: float
    index_se: float
    index_variance: float
    general_z: float
    conditional_se_z: float
    family_sensitivity_se_z: float
    family_count: int
    effective_evidence: float
    coverage: float
    confidence: str
    calibration_id: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _confidence(family_count: int, half_width: float) -> str:
    if family_count >= 12 and half_width <= 5.0:
        return "mature"
    if family_count >= 8 and half_width <= 7.5:
        return "established"
    return "fresh"


def _projection(
    rows: list[tuple[BenchmarkObservation, float]],
    benchmark_by_id: dict[str, BenchmarkDefinition],
    calibration: dict[str, object],
    *,
    omitted_family: str | None,
    ridge: float,
) -> float | None:
    numerator = 0.0
    denominator = ridge
    used = 0
    for observation, weight in rows:
        benchmark = benchmark_by_id[observation.benchmark_id]
        calibrated = calibration[observation.benchmark_id]
        family_id = calibrated.family_id
        if omitted_family is not None and family_id == omitted_family:
            continue
        score = normalize_score(observation.score, benchmark)
        loading = calibrated.general_loading_points_per_z / 10.0
        target = (score - calibrated.intercept_points) / 10.0
        numerator += weight * loading * target
        denominator += weight * loading * loading
        used += 1
    if used == 0:
        return None
    return numerator / max(denominator, 1e-12)


def _family_sensitivity_se(
    rows: list[tuple[BenchmarkObservation, float]],
    benchmark_by_id: dict[str, BenchmarkDefinition],
    calibration: dict[str, object],
    *,
    ridge: float,
) -> float:
    families = sorted(
        {
            calibration[observation.benchmark_id].family_id
            for observation, _ in rows
        }
    )
    if len(families) < 3:
        return 0.0
    values = [
        _projection(
            rows,
            benchmark_by_id,
            calibration,
            omitted_family=family_id,
            ridge=ridge,
        )
        for family_id in families
    ]
    array = np.asarray([value for value in values if value is not None], dtype=float)
    if len(array) < 2:
        return 0.0
    mean = float(array.mean())
    variance = (len(array) - 1) / len(array) * float(np.square(array - mean).sum())
    return math.sqrt(max(variance, 0.0))


def build_better_bench_index(
    models: list[ModelDefinition],
    benchmarks: list[BenchmarkDefinition],
    observations: list[BenchmarkObservation],
    estimator: EstimatorResult,
    adoption: list[BenchmarkAdoptionSnapshot] | None = None,
    *,
    as_of: date | None = None,
    calibration_id: str = "BBI-2026-09-04",
    center: float = 100.0,
    points_per_z: float = 10.0,
    ridge: float = 0.35,
) -> list[BetterBenchIndexEstimate]:
    """Convert the validated latent score to a versioned, uncertainty-aware index.

    BBI is a presentation transform, not a new estimator: 100 is the retained frontier
    cohort mean in the named calibration and ten index points equal one latent standard
    deviation. Variance combines the estimator's conditional uncertainty with fixed-
    calibration leave-one-benchmark-family-out sensitivity.
    """
    if points_per_z <= 0:
        raise ValueError("points_per_z must be positive")
    if ridge <= 0:
        raise ValueError("ridge must be positive")

    adoption = adoption or []
    as_of = as_of or date.today()
    retained_models = set(estimator.retained_models)
    retained_benchmarks = set(estimator.retained_benchmarks)
    benchmark_by_id = {benchmark.id: benchmark for benchmark in benchmarks}
    calibration = {row.benchmark_id: row for row in estimator.benchmarks}
    quality = {
        row.benchmark_id: row
        for row in rank_benchmarks(
            benchmarks,
            models,
            observations,
            adoption,
            as_of=as_of,
        )
    }

    rows_by_model: dict[str, list[tuple[BenchmarkObservation, float]]] = defaultdict(list)
    for observation in observations:
        if observation.model_id not in retained_models:
            continue
        if observation.benchmark_id not in retained_benchmarks:
            continue
        quality_row = quality.get(observation.benchmark_id)
        if quality_row is None:
            continue
        weight = (
            quality_row.family_adjusted_weight
            * _TIER_WEIGHT[quality_row.tier]
            * SOURCE_GRADE_WEIGHT[observation.source_grade]
        )
        if weight > 0:
            rows_by_model[observation.model_id].append((observation, float(weight)))

    estimates: list[BetterBenchIndexEstimate] = []
    for model in estimator.models:
        rows = rows_by_model.get(model.model_id, [])
        family_se = _family_sensitivity_se(
            rows,
            benchmark_by_id,
            calibration,
            ridge=ridge,
        )
        total_variance_z = model.conditional_se**2 + family_se**2
        total_se_z = math.sqrt(max(total_variance_z, 0.0))
        index_value = center + points_per_z * model.general_z
        index_se = points_per_z * total_se_z
        half_width = 1.96 * index_se
        estimates.append(
            BetterBenchIndexEstimate(
                model_id=model.model_id,
                index=round(index_value, 1),
                ci_low=round(index_value - half_width, 1),
                ci_high=round(index_value + half_width, 1),
                index_se=round(index_se, 2),
                index_variance=round(index_se**2, 2),
                general_z=model.general_z,
                conditional_se_z=round(model.conditional_se, 4),
                family_sensitivity_se_z=round(family_se, 4),
                family_count=model.family_count,
                effective_evidence=model.effective_evidence,
                coverage=model.coverage,
                confidence=_confidence(model.family_count, half_width),
                calibration_id=calibration_id,
            )
        )
    estimates.sort(key=lambda row: row.index, reverse=True)
    return estimates
