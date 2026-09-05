from __future__ import annotations

import math
from collections import defaultdict
from datetime import date

import numpy as np
import pandas as pd

from .schema import (
    SOURCE_GRADE_WEIGHT,
    BenchmarkDefinition,
    BenchmarkObservation,
    Capability,
    CapabilityScore,
    ModelDefinition,
    ModelScore,
)


def normalize_score(score: float, benchmark: BenchmarkDefinition) -> float:
    """Map a raw benchmark result to a fixed 0-100 capability scale."""
    span = benchmark.score_ceiling - benchmark.score_floor
    if benchmark.higher_is_better:
        value = (score - benchmark.score_floor) / span
    else:
        value = (benchmark.score_ceiling - score) / span
    return float(np.clip(value, 0.0, 1.0) * 100.0)


def freshness_weight(benchmark: BenchmarkDefinition, as_of: date) -> float:
    age_days = max((as_of - benchmark.published_at).days, 0)
    half_life_days = 36.0 * 30.4375
    return 0.65 + 0.35 * (0.5 ** (age_days / half_life_days))


def contamination_weight(
    benchmark: BenchmarkDefinition,
    model: ModelDefinition,
    as_of: date,
) -> float:
    """Estimate benchmark/model contamination risk from public exposure."""
    if benchmark.rotating or benchmark.sealed_test or benchmark.public_since is None:
        return 1.0
    exposure_end = model.training_cutoff or model.released_at or as_of
    exposed_days = (exposure_end - benchmark.public_since).days
    if exposed_days <= 0:
        return 1.0
    exposure_months = exposed_days / 30.4375
    return max(0.45, math.exp(-exposure_months / 48.0))


def estimate_redundancy_weights(
    observations: list[BenchmarkObservation],
    *,
    correlation_threshold: float = 0.80,
    minimum_overlap: int = 4,
) -> dict[str, float]:
    """Down-weight benchmark clusters that rank the same models almost identically."""
    if not observations:
        return {}
    frame = pd.DataFrame(
        [(o.model_id, o.benchmark_id, o.score) for o in observations],
        columns=["model_id", "benchmark_id", "score"],
    ).pivot_table(index="model_id", columns="benchmark_id", values="score", aggfunc="mean")
    ids = list(frame.columns)
    penalty = {benchmark_id: 0.0 for benchmark_id in ids}
    for i, left in enumerate(ids):
        for right in ids[i + 1 :]:
            pair = frame[[left, right]].dropna()
            if len(pair) < minimum_overlap:
                continue
            corr = pair[left].corr(pair[right], method="spearman")
            if pd.isna(corr):
                continue
            excess = max(0.0, abs(float(corr)) - correlation_threshold)
            if excess <= 0:
                continue
            scaled = excess / (1.0 - correlation_threshold)
            penalty[left] += scaled
            penalty[right] += scaled
    return {benchmark_id: 1.0 / (1.0 + value) for benchmark_id, value in penalty.items()}


def _weighted_summary(values: list[tuple[float, float]]) -> tuple[float, float, float]:
    arr = np.asarray([value for value, _ in values], dtype=float)
    weights = np.asarray([weight for _, weight in values], dtype=float)
    total_weight = float(weights.sum())
    if total_weight <= 0:
        raise ValueError("Evidence weights must be positive")
    mean = float(np.average(arr, weights=weights))
    n_eff = float((weights.sum() ** 2) / np.square(weights).sum())
    if len(arr) == 1:
        standard_error = 12.5
    else:
        variance = float(np.average(np.square(arr - mean), weights=weights))
        standard_error = max(math.sqrt(variance / max(n_eff, 1.0)), 3.0)
    return mean, standard_error, n_eff


def score_models(
    models: list[ModelDefinition],
    benchmarks: list[BenchmarkDefinition],
    observations: list[BenchmarkObservation],
    *,
    as_of: date | None = None,
    capability_weights: dict[Capability, float] | None = None,
) -> list[ModelScore]:
    """Compute transparent V0 capability profiles with uncertainty and coverage."""
    as_of = as_of or date.today()
    model_by_id = {model.id: model for model in models}
    benchmark_by_id = {benchmark.id: benchmark for benchmark in benchmarks}
    redundancy = estimate_redundancy_weights(observations)
    capability_weights = capability_weights or {capability: 1.0 for capability in Capability}
    evidence: dict[str, dict[Capability, list[tuple[float, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for observation in observations:
        model = model_by_id.get(observation.model_id)
        benchmark = benchmark_by_id.get(observation.benchmark_id)
        if model is None or benchmark is None:
            continue
        normalized = normalize_score(observation.score, benchmark)
        quality = (
            SOURCE_GRADE_WEIGHT[observation.source_grade]
            * benchmark.protocol_quality
            * benchmark.reliability
            * freshness_weight(benchmark, as_of)
            * contamination_weight(benchmark, model, as_of)
            * redundancy.get(benchmark.id, 1.0)
        )
        for capability, loading in benchmark.capability_loadings.items():
            if loading > 0:
                evidence[model.id][capability].append((normalized, quality * loading))
    total_domain_weight = sum(capability_weights.values())
    results: list[ModelScore] = []
    for model in models:
        cap_results: list[CapabilityScore] = []
        measured_for_general: list[tuple[float, float, float]] = []
        total_effective_benchmarks = 0.0
        observed_domain_weight = 0.0
        for capability in Capability:
            values = evidence[model.id].get(capability, [])
            if not values:
                cap_results.append(
                    CapabilityScore(
                        capability=capability,
                        score=None,
                        ci_low=None,
                        ci_high=None,
                        evidence_weight=0.0,
                        benchmark_count=0,
                    )
                )
                continue
            mean, se, n_eff = _weighted_summary(values)
            ci_half = 1.96 * se
            cap_results.append(
                CapabilityScore(
                    capability=capability,
                    score=round(mean, 2),
                    ci_low=round(max(0.0, mean - ci_half), 2),
                    ci_high=round(min(100.0, mean + ci_half), 2),
                    evidence_weight=round(sum(weight for _, weight in values), 4),
                    benchmark_count=len(values),
                )
            )
            domain_weight = capability_weights[capability]
            observed_domain_weight += domain_weight
            total_effective_benchmarks += n_eff
            measured_for_general.append((mean, se, domain_weight))
        coverage = observed_domain_weight / total_domain_weight if total_domain_weight else 0.0
        if not measured_for_general:
            general = low = high = None
        else:
            weight_sum = sum(weight for _, _, weight in measured_for_general)
            general = sum(mean * weight for mean, _, weight in measured_for_general) / weight_sum
            propagated_variance = sum(
                ((weight / weight_sum) ** 2) * (se**2)
                for _, se, weight in measured_for_general
            )
            missingness_sd = (1.0 - coverage) * 15.0
            total_se = math.sqrt(propagated_variance + missingness_sd**2)
            half = 1.96 * total_se
            low = max(0.0, general - half)
            high = min(100.0, general + half)
        results.append(
            ModelScore(
                model_id=model.id,
                model_name=model.name,
                general_score=round(general, 2) if general is not None else None,
                ci_low=round(low, 2) if low is not None else None,
                ci_high=round(high, 2) if high is not None else None,
                coverage=round(coverage, 4),
                effective_benchmarks=round(total_effective_benchmarks, 2),
                capability_scores=cap_results,
            )
        )
    return sorted(
        results,
        key=lambda item: item.general_score if item.general_score is not None else -1.0,
        reverse=True,
    )
