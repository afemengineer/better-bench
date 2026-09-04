from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import date
from enum import StrEnum

import numpy as np

from .schema import (
    BenchmarkAdoptionSnapshot,
    BenchmarkDefinition,
    BenchmarkObservation,
    ModelDefinition,
)
from .scoring import estimate_redundancy_weights, normalize_score


class BenchmarkTier(StrEnum):
    CORE = "core"
    EMERGING = "high_value_emerging"
    SUPPORTING = "supporting"
    DIAGNOSTIC = "diagnostic_only"


@dataclass(frozen=True)
class BenchmarkQualityRow:
    benchmark_id: str
    family_id: str
    tier: BenchmarkTier
    importance: float
    family_adjusted_weight: float
    quality: float
    adoption: float
    discrimination: float
    integrity: float
    independence: float
    observed_models: int
    effective_leaderboard_models: int
    observed_organizations: int


def _latest_adoption(
    rows: list[BenchmarkAdoptionSnapshot],
) -> dict[str, BenchmarkAdoptionSnapshot]:
    result: dict[str, BenchmarkAdoptionSnapshot] = {}
    for row in rows:
        current = result.get(row.benchmark_id)
        if current is None or row.as_of > current.as_of:
            result[row.benchmark_id] = row
    return result


def _integrity_score(benchmark: BenchmarkDefinition, as_of: date) -> float:
    if benchmark.contamination_resistance is not None:
        return benchmark.contamination_resistance
    if benchmark.sealed_test:
        return 1.0
    if benchmark.rotating:
        return 0.95
    age_days = max((as_of - benchmark.published_at).days, 0)
    if age_days <= 90:
        return 0.85
    if age_days <= 365:
        return 0.70
    return 0.50


def _discrimination_score(
    benchmark: BenchmarkDefinition,
    rows: list[BenchmarkObservation],
) -> float:
    values = [
        normalize_score(row.score, benchmark)
        for row in rows
        if row.benchmark_id == benchmark.id
    ]
    if len(values) < 3:
        return 0.20
    arr = np.asarray(values, dtype=float)
    spread = min(float(arr.std(ddof=0)) / 15.0, 1.0)
    median = float(np.median(arr))
    headroom = min(min(median, 100.0 - median) / 20.0, 1.0)
    return float(math.sqrt(max(spread, 0.02) * max(headroom, 0.02)))


def _adoption_score(model_count: int, org_count: int) -> float:
    model_component = min(math.log1p(model_count) / math.log1p(24), 1.0)
    org_component = min(org_count / 6.0, 1.0)
    return 0.70 * model_component + 0.30 * org_component


def _geometric_importance(components: dict[str, float]) -> float:
    weights = {
        "quality": 0.30,
        "adoption": 0.20,
        "discrimination": 0.25,
        "integrity": 0.15,
        "independence": 0.10,
    }
    return math.exp(
        sum(weights[name] * math.log(max(components[name], 0.05)) for name in weights)
    )


def _tier(
    *,
    importance: float,
    quality: float,
    adoption: float,
    discrimination: float,
    integrity: float,
    model_count: int,
) -> BenchmarkTier:
    if quality < 0.60 or discrimination < 0.18:
        return BenchmarkTier.DIAGNOSTIC
    if (
        importance >= 0.68
        and adoption >= 0.55
        and model_count >= 10
        and quality >= 0.75
        and discrimination >= 0.30
    ):
        return BenchmarkTier.CORE
    if (
        model_count < 10
        and quality >= 0.80
        and discrimination >= 0.30
        and integrity >= 0.70
    ):
        return BenchmarkTier.EMERGING
    if importance >= 0.50:
        return BenchmarkTier.SUPPORTING
    return BenchmarkTier.DIAGNOSTIC


def _apply_family_budget(rows: list[BenchmarkQualityRow]) -> list[BenchmarkQualityRow]:
    """Give a benchmark family one evidence budget instead of one vote per subdivision.

    A family's total adjusted weight equals the strongest member's raw importance.
    Members split that budget in proportion to their raw importance. A singleton is
    unchanged. This prevents suites such as LiveBench from gaining seven times the
    overall influence simply because they publish seven category scores.
    """
    by_family: dict[str, list[BenchmarkQualityRow]] = defaultdict(list)
    for row in rows:
        by_family[row.family_id].append(row)

    adjusted: list[BenchmarkQualityRow] = []
    for family_rows in by_family.values():
        total = sum(row.importance for row in family_rows)
        budget = max(row.importance for row in family_rows)
        for row in family_rows:
            weight = budget * row.importance / total if total > 0 else 0.0
            adjusted.append(replace(row, family_adjusted_weight=round(weight, 4)))
    return adjusted


def rank_benchmarks(
    benchmarks: list[BenchmarkDefinition],
    models: list[ModelDefinition],
    observations: list[BenchmarkObservation],
    adoption: list[BenchmarkAdoptionSnapshot] | None = None,
    *,
    as_of: date | None = None,
) -> list[BenchmarkQualityRow]:
    """Rank benchmark evidence utility without letting popularity dominate quality."""
    as_of = as_of or date.today()
    adoption_by_id = _latest_adoption(adoption or [])
    model_by_id = {model.id: model for model in models}
    redundancy = estimate_redundancy_weights(observations)

    observed_models_by_benchmark: dict[str, set[str]] = {}
    observed_orgs_by_benchmark: dict[str, set[str]] = {}
    for row in observations:
        observed_models_by_benchmark.setdefault(row.benchmark_id, set()).add(row.model_id)
        model = model_by_id.get(row.model_id)
        if model is not None and model.organization:
            observed_orgs_by_benchmark.setdefault(row.benchmark_id, set()).add(
                model.organization
            )

    results: list[BenchmarkQualityRow] = []
    for benchmark in benchmarks:
        observed_models = len(observed_models_by_benchmark.get(benchmark.id, set()))
        observed_orgs = len(observed_orgs_by_benchmark.get(benchmark.id, set()))
        external = adoption_by_id.get(benchmark.id)
        model_count = max(
            observed_models,
            external.leaderboard_model_count if external and external.leaderboard_model_count else 0,
        )
        org_count = max(
            observed_orgs,
            external.leaderboard_org_count if external and external.leaderboard_org_count else 0,
        )
        quality = math.sqrt(benchmark.protocol_quality * benchmark.reliability)
        adoption_score = _adoption_score(model_count, org_count)
        discrimination = _discrimination_score(benchmark, observations)
        integrity = _integrity_score(benchmark, as_of)
        independence = redundancy.get(benchmark.id, 1.0)
        importance = _geometric_importance(
            {
                "quality": quality,
                "adoption": adoption_score,
                "discrimination": discrimination,
                "integrity": integrity,
                "independence": independence,
            }
        )
        results.append(
            BenchmarkQualityRow(
                benchmark_id=benchmark.id,
                family_id=benchmark.family_id or benchmark.id,
                tier=_tier(
                    importance=importance,
                    quality=quality,
                    adoption=adoption_score,
                    discrimination=discrimination,
                    integrity=integrity,
                    model_count=model_count,
                ),
                importance=round(importance, 4),
                family_adjusted_weight=round(importance, 4),
                quality=round(quality, 4),
                adoption=round(adoption_score, 4),
                discrimination=round(discrimination, 4),
                integrity=round(integrity, 4),
                independence=round(independence, 4),
                observed_models=observed_models,
                effective_leaderboard_models=model_count,
                observed_organizations=org_count,
            )
        )
    results = _apply_family_budget(results)
    tier_order = {
        BenchmarkTier.CORE: 0,
        BenchmarkTier.EMERGING: 1,
        BenchmarkTier.SUPPORTING: 2,
        BenchmarkTier.DIAGNOSTIC: 3,
    }
    return sorted(results, key=lambda row: (tier_order[row.tier], -row.importance))
