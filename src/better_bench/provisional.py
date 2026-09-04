from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import date

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
class ProvisionalEstimate:
    model_id: str
    projected_general_z: float | None
    benchmark_count: int
    family_count: int
    effective_evidence: float
    coverage: float
    status: str
    confidence: str
    reason: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _effective_family_evidence(family_weights: dict[str, float]) -> float:
    values = list(family_weights.values())
    total = sum(values)
    denominator = sum(value * value for value in values)
    if denominator <= 0:
        return 0.0
    return total * total / denominator


def _confidence(family_count: int, minimum_families: int) -> str:
    if family_count <= 1:
        return "very_low"
    if family_count <= 3:
        return "low"
    if family_count < minimum_families:
        return "near_rankable"
    return "structural_overlap_missing"


def project_provisional_models(
    models: list[ModelDefinition],
    benchmarks: list[BenchmarkDefinition],
    observations: list[BenchmarkObservation],
    estimator: EstimatorResult,
    adoption: list[BenchmarkAdoptionSnapshot] | None = None,
    *,
    as_of: date | None = None,
    minimum_families: int = 5,
    ridge: float = 0.35,
) -> list[ProvisionalEstimate]:
    """Project sparse models onto the calibrated general factor without refitting it.

    Provisional models never influence benchmark intercepts, loadings, evidence weights,
    or the official ranking. Their score is a ridge-shrunk projection from whatever
    retained benchmark evidence exists. It is intentionally labelled non-rankable.
    """
    if minimum_families < 1:
        raise ValueError("minimum_families must be at least 1")
    if ridge <= 0:
        raise ValueError("ridge must be positive")

    adoption = adoption or []
    as_of = as_of or date.today()
    retained_models = set(estimator.retained_models)
    calibration = {row.benchmark_id: row for row in estimator.benchmarks}
    retained_benchmarks = set(calibration)
    benchmark_by_id = {row.id: row for row in benchmarks}
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
    total_families = len({row.family_id for row in estimator.benchmarks})

    by_model: dict[str, list[tuple[BenchmarkObservation, float]]] = defaultdict(list)
    for observation in observations:
        if observation.model_id in retained_models:
            continue
        if observation.benchmark_id not in retained_benchmarks:
            continue
        benchmark = benchmark_by_id.get(observation.benchmark_id)
        quality_row = quality.get(observation.benchmark_id)
        if benchmark is None or quality_row is None:
            continue
        weight = (
            quality_row.family_adjusted_weight
            * _TIER_WEIGHT[quality_row.tier]
            * SOURCE_GRADE_WEIGHT[observation.source_grade]
        )
        if weight <= 0:
            continue
        by_model[observation.model_id].append((observation, float(weight)))

    results: list[ProvisionalEstimate] = []
    for model in models:
        if model.id in retained_models:
            continue
        rows = by_model.get(model.id, [])
        numerator = 0.0
        denominator = ridge
        family_weights: dict[str, float] = defaultdict(float)
        benchmarks_seen: set[str] = set()
        families_seen: set[str] = set()
        for observation, weight in rows:
            benchmark = benchmark_by_id[observation.benchmark_id]
            calibrated = calibration[observation.benchmark_id]
            score = normalize_score(observation.score, benchmark)
            loading = calibrated.general_loading_points_per_z / 10.0
            target = (score - calibrated.intercept_points) / 10.0
            numerator += weight * loading * target
            denominator += weight * loading * loading
            family_id = calibrated.family_id
            family_weights[family_id] += weight
            benchmarks_seen.add(observation.benchmark_id)
            families_seen.add(family_id)

        family_count = len(families_seen)
        if rows:
            projected = numerator / max(denominator, 1e-12)
            projected_value: float | None = round(float(projected), 4)
        else:
            projected_value = None

        if not rows:
            status = "unscored"
            reason = "no observations overlap the calibrated benchmark set"
        elif family_count < minimum_families:
            status = "provisional"
            missing = minimum_families - family_count
            noun = "family" if missing == 1 else "families"
            reason = f"needs {missing} more independent benchmark {noun}"
        else:
            status = "provisional"
            reason = (
                "meets raw family count but failed the iterative model-benchmark "
                "overlap filter"
            )

        results.append(
            ProvisionalEstimate(
                model_id=model.id,
                projected_general_z=projected_value,
                benchmark_count=len(benchmarks_seen),
                family_count=family_count,
                effective_evidence=round(_effective_family_evidence(family_weights), 3),
                coverage=round(family_count / max(total_families, 1), 4),
                status=status,
                confidence=_confidence(family_count, minimum_families),
                reason=reason,
            )
        )

    results.sort(
        key=lambda row: (
            row.projected_general_z is not None,
            row.projected_general_z if row.projected_general_z is not None else -math.inf,
        ),
        reverse=True,
    )
    return results
