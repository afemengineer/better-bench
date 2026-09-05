from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import date

import numpy as np

from .benchmark_quality import BenchmarkQualityRow, rank_benchmarks
from .schema import (
    SOURCE_GRADE_WEIGHT,
    BenchmarkAdoptionSnapshot,
    BenchmarkDefinition,
    BenchmarkObservation,
    ModelDefinition,
)
from .scoring import normalize_score


@dataclass(frozen=True)
class RankingObservation:
    model_id: str
    benchmark_id: str
    family_id: str
    score_points: float
    weight: float


@dataclass(frozen=True)
class RankingEvidenceSummary:
    observations: list[RankingObservation]
    benchmark_weights: dict[str, float]
    retained_benchmarks: list[str]
    retained_families: list[str]


def _ordinal_importance(row: BenchmarkQualityRow) -> float:
    """Evidence importance for ranking without fitting to the current leaderboard.

    The score-regression estimator needs empirical score discrimination to learn a
    numerical loading. An ordinal comparison does not: benchmark difficulty cancels in
    a head-to-head comparison, and even a benchmark with only two frontier results can
    provide high-value ordering evidence. We therefore use protocol/reliability quality,
    contamination integrity, independence, and a deliberately weak adoption term.

    The adoption floor prevents a brand-new but rigorous benchmark from being assigned
    near-zero weight merely because few models have been evaluated yet. Adoption still
    matters, but it cannot dominate scientific quality or create a rich-get-richer loop.
    """

    components = {
        "quality": max(float(row.quality), 0.05),
        "integrity": max(float(row.integrity), 0.05),
        "independence": max(float(row.independence), 0.05),
        "adoption": max(float(row.adoption), 0.25),
    }
    powers = {
        "quality": 0.40,
        "integrity": 0.25,
        "independence": 0.20,
        "adoption": 0.15,
    }
    return math.exp(
        sum(powers[name] * math.log(components[name]) for name in powers)
    )


def prepare_ranking_evidence(
    models: list[ModelDefinition],
    benchmarks: list[BenchmarkDefinition],
    observations: list[BenchmarkObservation],
    adoption: list[BenchmarkAdoptionSnapshot],
    *,
    rankable_model_ids: list[str],
    as_of: date,
    minimum_models_per_benchmark: int = 2,
) -> RankingEvidenceSummary:
    """Build the broad evidence panel used only for final model ordering.

    Rankability remains determined by the conservative calibration estimator. Once a
    model is rankable, however, *all* revision-safe benchmark evidence shared with at
    least one other rankable model can inform ordering. This intentionally decouples the
    >=5-model requirement needed to estimate benchmark intercept/loadings from the much
    weaker >=2-model requirement needed for a within-benchmark comparison.

    Family budgets are recomputed after eligibility filtering so an unusable one-model
    protocol cannot consume the budget of a related benchmark that actually compares
    frontier models.
    """

    if minimum_models_per_benchmark < 2:
        raise ValueError("minimum_models_per_benchmark must be at least 2")
    rankable = set(rankable_model_ids)
    benchmark_by_id = {benchmark.id: benchmark for benchmark in benchmarks}
    quality_rows = rank_benchmarks(
        benchmarks,
        models,
        observations,
        adoption,
        as_of=as_of,
    )
    quality_by_id = {row.benchmark_id: row for row in quality_rows}

    observed_rankable: dict[str, set[str]] = defaultdict(set)
    for observation in observations:
        if observation.model_id in rankable:
            observed_rankable[observation.benchmark_id].add(observation.model_id)

    eligible = {
        benchmark_id
        for benchmark_id, model_ids in observed_rankable.items()
        if len(model_ids) >= minimum_models_per_benchmark
        and benchmark_id in benchmark_by_id
        and benchmark_id in quality_by_id
    }
    if not eligible:
        raise ValueError("No pairwise-comparable ranking benchmarks remain")

    raw_importance = {
        benchmark_id: _ordinal_importance(quality_by_id[benchmark_id])
        for benchmark_id in eligible
    }
    by_family: dict[str, list[str]] = defaultdict(list)
    for benchmark_id in eligible:
        benchmark = benchmark_by_id[benchmark_id]
        by_family[benchmark.family_id or benchmark.id].append(benchmark_id)

    benchmark_weight: dict[str, float] = {}
    for family_ids in by_family.values():
        family_total = sum(raw_importance[benchmark_id] for benchmark_id in family_ids)
        family_budget = max(raw_importance[benchmark_id] for benchmark_id in family_ids)
        for benchmark_id in family_ids:
            benchmark_weight[benchmark_id] = (
                family_budget * raw_importance[benchmark_id] / family_total
                if family_total > 0
                else 0.0
            )

    rows: list[RankingObservation] = []
    for observation in observations:
        if observation.model_id not in rankable or observation.benchmark_id not in eligible:
            continue
        benchmark = benchmark_by_id[observation.benchmark_id]
        source_weight = SOURCE_GRADE_WEIGHT[observation.source_grade]
        rows.append(
            RankingObservation(
                model_id=observation.model_id,
                benchmark_id=observation.benchmark_id,
                family_id=benchmark.family_id or benchmark.id,
                score_points=float(normalize_score(observation.score, benchmark)),
                weight=max(
                    float(benchmark_weight[observation.benchmark_id] * source_weight),
                    1e-6,
                ),
            )
        )
    if not rows:
        raise ValueError("No ranking observations remain after eligibility filtering")

    mean_weight = float(np.mean([row.weight for row in rows]))
    rows = [
        RankingObservation(
            model_id=row.model_id,
            benchmark_id=row.benchmark_id,
            family_id=row.family_id,
            score_points=row.score_points,
            weight=row.weight / max(mean_weight, 1e-12),
        )
        for row in rows
    ]
    return RankingEvidenceSummary(
        observations=rows,
        benchmark_weights=benchmark_weight,
        retained_benchmarks=sorted(eligible),
        retained_families=sorted(by_family),
    )
