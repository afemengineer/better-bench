from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import date

import numpy as np
from scipy.stats import spearmanr

from .benchmark_quality import BenchmarkTier, rank_benchmarks
from .factors import _filter_matrix, _normalized_matrix
from .schema import (
    SOURCE_GRADE_WEIGHT,
    BenchmarkAdoptionSnapshot,
    BenchmarkDefinition,
    BenchmarkObservation,
    Capability,
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
class EstimatorConfig:
    minimum_models_per_benchmark: int = 5
    minimum_benchmarks_per_model: int = 5
    minimum_families_per_model: int = 5
    score_unit_points: float = 10.0
    ridge_general: float = 0.35
    ridge_loading: float = 1.5
    loading_minimum: float = 0.05
    loading_maximum: float = 3.0
    domain_ridge: float = 4.0
    maximum_iterations: int = 250
    tolerance: float = 1e-7
    jackknife_uncertainty: bool = True


@dataclass(frozen=True)
class GeneralEstimate:
    model_id: str
    general_z: float
    ci_low: float
    ci_high: float
    conditional_se: float
    family_jackknife_se: float
    benchmark_count: int
    family_count: int
    effective_evidence: float
    coverage: float
    domain_residual_points: dict[str, float]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class BenchmarkCalibration:
    benchmark_id: str
    family_id: str
    tier: str
    evidence_weight: float
    intercept_points: float
    general_loading_points_per_z: float
    observed_models: int


@dataclass(frozen=True)
class EstimatorResult:
    models: list[GeneralEstimate]
    benchmarks: list[BenchmarkCalibration]
    retained_models: list[str]
    retained_benchmarks: list[str]
    observed_cells: int
    possible_cells: int
    density: float
    weighted_rmse_points: float
    weighted_r2_vs_benchmark_mean: float
    iterations: int
    converged: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "fit": {
                "retained_models": len(self.retained_models),
                "retained_benchmarks": len(self.retained_benchmarks),
                "observed_cells": self.observed_cells,
                "possible_cells": self.possible_cells,
                "density": self.density,
                "weighted_rmse_points": self.weighted_rmse_points,
                "weighted_r2_vs_benchmark_mean": self.weighted_r2_vs_benchmark_mean,
                "iterations": self.iterations,
                "converged": self.converged,
            },
            "models": [row.to_dict() for row in self.models],
            "benchmarks": [asdict(row) for row in self.benchmarks],
        }


@dataclass(frozen=True)
class EstimatorCrossValidation:
    folds: int
    heldout_observations: int
    model_rmse_points: float
    benchmark_only_rmse_points: float
    relative_rmse_improvement: float
    residual_r2: float
    residual_spearman: float


@dataclass(frozen=True)
class _Observation:
    model_id: str
    benchmark_id: str
    family_id: str
    score_points: float
    weight: float


@dataclass
class _State:
    general: dict[str, float]
    intercept: dict[str, float]
    loading: dict[str, float]


def _effective_evidence(rows: list[_Observation]) -> float:
    by_family: dict[str, float] = defaultdict(float)
    for row in rows:
        by_family[row.family_id] += row.weight
    values = np.asarray(list(by_family.values()), dtype=float)
    denominator = float(np.square(values).sum())
    if denominator <= 0:
        return 0.0
    return float(values.sum() ** 2 / denominator)


def _prepare(
    models: list[ModelDefinition],
    benchmarks: list[BenchmarkDefinition],
    observations: list[BenchmarkObservation],
    adoption: list[BenchmarkAdoptionSnapshot],
    config: EstimatorConfig,
    as_of: date,
) -> tuple[
    list[_Observation],
    list[str],
    list[str],
    dict[str, BenchmarkDefinition],
    dict[str, float],
    dict[str, str],
]:
    if config.score_unit_points <= 0:
        raise ValueError("score_unit_points must be positive")
    benchmark_by_id = {row.id: row for row in benchmarks}
    families = {row.id: row.family_id or row.id for row in benchmarks}
    frame = _filter_matrix(
        _normalized_matrix(benchmarks, observations),
        config.minimum_models_per_benchmark,
        config.minimum_benchmarks_per_model,
        families,
        config.minimum_families_per_model,
    )
    if frame.empty:
        raise ValueError("Insufficient overlapping data for estimator")

    model_ids = [str(item) for item in frame.index]
    benchmark_ids = [str(item) for item in frame.columns]
    retained_models = set(model_ids)
    retained_benchmarks = set(benchmark_ids)
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

    rows: list[_Observation] = []
    for observation in observations:
        if (
            observation.model_id not in retained_models
            or observation.benchmark_id not in retained_benchmarks
        ):
            continue
        benchmark = benchmark_by_id.get(observation.benchmark_id)
        quality_row = quality.get(observation.benchmark_id)
        if benchmark is None or quality_row is None:
            continue
        score = normalize_score(observation.score, benchmark)
        benchmark_weight = (
            quality_row.family_adjusted_weight * _TIER_WEIGHT[quality_row.tier]
        )
        source_weight = SOURCE_GRADE_WEIGHT[observation.source_grade]
        rows.append(
            _Observation(
                model_id=observation.model_id,
                benchmark_id=observation.benchmark_id,
                family_id=benchmark.family_id or benchmark.id,
                score_points=float(score),
                weight=max(float(benchmark_weight * source_weight), 1e-5),
            )
        )
    if not rows:
        raise ValueError("No observations remained after estimator filtering")

    mean_weight = float(np.mean([row.weight for row in rows]))
    rows = [
        _Observation(
            model_id=row.model_id,
            benchmark_id=row.benchmark_id,
            family_id=row.family_id,
            score_points=row.score_points,
            weight=row.weight / mean_weight,
        )
        for row in rows
    ]
    evidence_weight = {
        benchmark_id: quality[benchmark_id].family_adjusted_weight
        for benchmark_id in benchmark_ids
    }
    tiers = {
        benchmark_id: quality[benchmark_id].tier.value for benchmark_id in benchmark_ids
    }
    return (
        rows,
        model_ids,
        benchmark_ids,
        benchmark_by_id,
        evidence_weight,
        tiers,
    )


def _group_rows(
    rows: list[_Observation],
) -> tuple[dict[str, list[_Observation]], dict[str, list[_Observation]]]:
    by_model: dict[str, list[_Observation]] = defaultdict(list)
    by_benchmark: dict[str, list[_Observation]] = defaultdict(list)
    for row in rows:
        by_model[row.model_id].append(row)
        by_benchmark[row.benchmark_id].append(row)
    return by_model, by_benchmark


def _fit_state(
    rows: list[_Observation],
    model_ids: list[str],
    benchmark_ids: list[str],
    config: EstimatorConfig,
    *,
    family_multipliers: dict[str, float] | None = None,
) -> tuple[_State, int, bool]:
    by_model, by_benchmark = _group_rows(rows)
    multipliers = family_multipliers or {}

    intercept: dict[str, float] = {}
    for benchmark_id in benchmark_ids:
        group = by_benchmark.get(benchmark_id, [])
        if not group:
            intercept[benchmark_id] = 50.0 / config.score_unit_points
            continue
        weights = np.asarray(
            [row.weight * multipliers.get(row.family_id, 1.0) for row in group],
            dtype=float,
        )
        values = np.asarray(
            [row.score_points / config.score_unit_points for row in group], dtype=float
        )
        if float(weights.sum()) <= 1e-12:
            intercept[benchmark_id] = 50.0 / config.score_unit_points
            continue
        intercept[benchmark_id] = float(np.average(values, weights=weights))

    state = _State(
        general={model_id: 0.0 for model_id in model_ids},
        intercept=intercept,
        loading={benchmark_id: 1.0 for benchmark_id in benchmark_ids},
    )

    previous = float("inf")
    converged = False
    iteration = 0
    for iteration in range(1, config.maximum_iterations + 1):
        for model_id in model_ids:
            numerator = 0.0
            denominator = config.ridge_general
            for row in by_model.get(model_id, []):
                weight = row.weight * multipliers.get(row.family_id, 1.0)
                loading = state.loading[row.benchmark_id]
                target = (
                    row.score_points / config.score_unit_points
                    - state.intercept[row.benchmark_id]
                )
                numerator += weight * loading * target
                denominator += weight * loading * loading
            state.general[model_id] = numerator / max(denominator, 1e-12)

        general_mean = float(np.mean(list(state.general.values())))
        if abs(general_mean) > 1e-12:
            for model_id in model_ids:
                state.general[model_id] -= general_mean
            for benchmark_id in benchmark_ids:
                state.intercept[benchmark_id] += (
                    state.loading[benchmark_id] * general_mean
                )

        for benchmark_id in benchmark_ids:
            group = by_benchmark.get(benchmark_id, [])
            if not group:
                continue
            weight_sum = 0.0
            residual_sum = 0.0
            for row in group:
                weight = row.weight * multipliers.get(row.family_id, 1.0)
                residual = (
                    row.score_points / config.score_unit_points
                    - state.loading[benchmark_id] * state.general[row.model_id]
                )
                residual_sum += weight * residual
                weight_sum += weight
            if weight_sum <= 1e-12:
                continue
            state.intercept[benchmark_id] = residual_sum / weight_sum

        for benchmark_id in benchmark_ids:
            numerator = config.ridge_loading
            denominator = config.ridge_loading
            active_weight = 0.0
            for row in by_benchmark.get(benchmark_id, []):
                weight = row.weight * multipliers.get(row.family_id, 1.0)
                active_weight += weight
                general = state.general[row.model_id]
                target = (
                    row.score_points / config.score_unit_points
                    - state.intercept[benchmark_id]
                )
                numerator += weight * general * target
                denominator += weight * general * general
            if active_weight <= 1e-12:
                continue
            state.loading[benchmark_id] = float(
                np.clip(
                    numerator / max(denominator, 1e-12),
                    config.loading_minimum,
                    config.loading_maximum,
                )
            )

        loss = 0.0
        for row in rows:
            weight = row.weight * multipliers.get(row.family_id, 1.0)
            prediction = state.intercept[row.benchmark_id] + (
                state.loading[row.benchmark_id] * state.general[row.model_id]
            )
            residual = row.score_points / config.score_unit_points - prediction
            loss += weight * residual * residual
        if previous < float("inf"):
            relative = abs(previous - loss) / max(previous, 1e-12)
            if relative < config.tolerance:
                converged = True
                break
        previous = loss
    return state, iteration, converged


def _general_scale(state: _State, model_ids: list[str]) -> tuple[float, float]:
    values = np.asarray([state.general[model_id] for model_id in model_ids], dtype=float)
    mean = float(values.mean())
    scale = float(values.std(ddof=0))
    if scale <= 1e-10:
        raise ValueError("General factor collapsed to zero variance")
    return mean, scale


def _domain_profiles(
    rows: list[_Observation],
    state: _State,
    model_ids: list[str],
    benchmark_by_id: dict[str, BenchmarkDefinition],
    config: EstimatorConfig,
) -> dict[str, dict[str, float]]:
    capabilities = list(Capability)
    vectors: dict[str, np.ndarray] = {}
    for benchmark_id, benchmark in benchmark_by_id.items():
        vectors[benchmark_id] = np.asarray(
            [benchmark.capability_loadings.get(capability, 0.0) for capability in capabilities],
            dtype=float,
        )
    retained_vectors = [vectors[row.benchmark_id] for row in rows]
    center = np.mean(np.vstack(retained_vectors), axis=0)
    by_model, _ = _group_rows(rows)
    profiles: dict[str, dict[str, float]] = {}
    penalty = config.domain_ridge * np.eye(len(capabilities))
    for model_id in model_ids:
        group = by_model.get(model_id, [])
        if not group:
            profiles[model_id] = {capability.value: 0.0 for capability in capabilities}
            continue
        design = np.vstack([vectors[row.benchmark_id] - center for row in group])
        target = np.asarray(
            [
                row.score_points
                - config.score_unit_points
                * (
                    state.intercept[row.benchmark_id]
                    + state.loading[row.benchmark_id] * state.general[model_id]
                )
                for row in group
            ],
            dtype=float,
        )
        weights = np.asarray([row.weight for row in group], dtype=float)
        xtw = design.T * weights
        beta = np.linalg.solve(xtw @ design + penalty, xtw @ target)
        beta -= float(beta.mean())
        profiles[model_id] = {
            capability.value: round(float(beta[index]), 3)
            for index, capability in enumerate(capabilities)
        }
    return profiles


def _conditional_se(
    rows: list[_Observation],
    state: _State,
    model_id: str,
    residual_variance: float,
    general_scale: float,
    config: EstimatorConfig,
) -> float:
    information = config.ridge_general
    for row in rows:
        if row.model_id == model_id:
            information += row.weight * state.loading[row.benchmark_id] ** 2
    raw_se = math.sqrt(residual_variance / max(information, 1e-12))
    return raw_se / general_scale


def _family_jackknife(
    rows: list[_Observation],
    model_ids: list[str],
    benchmark_ids: list[str],
    config: EstimatorConfig,
) -> dict[str, float]:
    families = sorted({row.family_id for row in rows})
    if len(families) < 3:
        return {model_id: 0.0 for model_id in model_ids}
    estimates: dict[str, list[float]] = defaultdict(list)
    for omitted in families:
        multipliers = {omitted: 0.0}
        state, _, _ = _fit_state(
            rows,
            model_ids,
            benchmark_ids,
            config,
            family_multipliers=multipliers,
        )
        mean, scale = _general_scale(state, model_ids)
        for model_id in model_ids:
            estimates[model_id].append((state.general[model_id] - mean) / scale)

    result: dict[str, float] = {}
    count = len(families)
    for model_id in model_ids:
        values = np.asarray(estimates[model_id], dtype=float)
        mean = float(values.mean())
        variance = (count - 1) / count * float(np.square(values - mean).sum())
        result[model_id] = math.sqrt(max(variance, 0.0))
    return result


def fit_estimator(
    models: list[ModelDefinition],
    benchmarks: list[BenchmarkDefinition],
    observations: list[BenchmarkObservation],
    adoption: list[BenchmarkAdoptionSnapshot] | None = None,
    *,
    config: EstimatorConfig | None = None,
    as_of: date | None = None,
) -> EstimatorResult:
    """Fit the validated Better Bench general-capability estimator.

    The ranking core is intentionally conservative: normalized benchmark percentages are
    kept on a fixed scale, benchmark difficulty is represented by an intercept, frontier
    discrimination by a learned loading, and evidence influence by benchmark-family and
    provenance weights. Capability residuals are reported descriptively and do not feed
    the general score until they demonstrate held-out predictive value.
    """
    config = config or EstimatorConfig()
    as_of = as_of or date.today()
    adoption = adoption or []
    (
        rows,
        model_ids,
        benchmark_ids,
        benchmark_by_id,
        evidence_weight,
        tiers,
    ) = _prepare(models, benchmarks, observations, adoption, config, as_of)
    state, iterations, converged = _fit_state(rows, model_ids, benchmark_ids, config)
    general_mean, general_scale = _general_scale(state, model_ids)

    predictions = np.asarray(
        [
            config.score_unit_points
            * (
                state.intercept[row.benchmark_id]
                + state.loading[row.benchmark_id] * state.general[row.model_id]
            )
            for row in rows
        ],
        dtype=float,
    )
    observed = np.asarray([row.score_points for row in rows], dtype=float)
    weights = np.asarray([row.weight for row in rows], dtype=float)
    residual = observed - predictions
    weighted_sse = float(np.sum(weights * np.square(residual)))
    residual_variance_units = (
        weighted_sse / max(float(weights.sum()), 1.0) / config.score_unit_points**2
    )
    weighted_rmse = math.sqrt(weighted_sse / max(float(weights.sum()), 1e-12))

    benchmark_only = np.asarray(
        [config.score_unit_points * state.intercept[row.benchmark_id] for row in rows],
        dtype=float,
    )
    baseline_sse = float(np.sum(weights * np.square(observed - benchmark_only)))
    r2 = 1.0 - weighted_sse / baseline_sse if baseline_sse > 1e-12 else 0.0

    by_model, by_benchmark = _group_rows(rows)
    domain_profiles = _domain_profiles(
        rows,
        state,
        model_ids,
        benchmark_by_id,
        config,
    )
    jackknife = (
        _family_jackknife(rows, model_ids, benchmark_ids, config)
        if config.jackknife_uncertainty
        else {model_id: 0.0 for model_id in model_ids}
    )
    total_families = len({row.family_id for row in rows})

    model_estimates: list[GeneralEstimate] = []
    for model_id in model_ids:
        model_rows = by_model.get(model_id, [])
        general_z = (state.general[model_id] - general_mean) / general_scale
        conditional = _conditional_se(
            model_rows,
            state,
            model_id,
            residual_variance_units,
            general_scale,
            config,
        )
        family_se = jackknife.get(model_id, 0.0)
        total_se = math.sqrt(conditional**2 + family_se**2)
        families = {row.family_id for row in model_rows}
        model_estimates.append(
            GeneralEstimate(
                model_id=model_id,
                general_z=round(float(general_z), 4),
                ci_low=round(float(general_z - 1.96 * total_se), 4),
                ci_high=round(float(general_z + 1.96 * total_se), 4),
                conditional_se=round(float(conditional), 4),
                family_jackknife_se=round(float(family_se), 4),
                benchmark_count=len({row.benchmark_id for row in model_rows}),
                family_count=len(families),
                effective_evidence=round(_effective_evidence(model_rows), 3),
                coverage=round(len(families) / max(total_families, 1), 4),
                domain_residual_points=domain_profiles[model_id],
            )
        )
    model_estimates.sort(key=lambda row: row.general_z, reverse=True)

    benchmark_estimates = [
        BenchmarkCalibration(
            benchmark_id=benchmark_id,
            family_id=benchmark_by_id[benchmark_id].family_id or benchmark_id,
            tier=tiers[benchmark_id],
            evidence_weight=round(evidence_weight[benchmark_id], 4),
            intercept_points=round(
                config.score_unit_points * state.intercept[benchmark_id], 3
            ),
            general_loading_points_per_z=round(
                config.score_unit_points * state.loading[benchmark_id] * general_scale,
                3,
            ),
            observed_models=len({row.model_id for row in by_benchmark[benchmark_id]}),
        )
        for benchmark_id in benchmark_ids
    ]
    benchmark_estimates.sort(
        key=lambda row: abs(row.general_loading_points_per_z), reverse=True
    )

    observed_cells = len({(row.model_id, row.benchmark_id) for row in rows})
    possible_cells = len(model_ids) * len(benchmark_ids)
    return EstimatorResult(
        models=model_estimates,
        benchmarks=benchmark_estimates,
        retained_models=model_ids,
        retained_benchmarks=benchmark_ids,
        observed_cells=observed_cells,
        possible_cells=possible_cells,
        density=round(observed_cells / possible_cells, 4),
        weighted_rmse_points=round(weighted_rmse, 3),
        weighted_r2_vs_benchmark_mean=round(r2, 4),
        iterations=iterations,
        converged=converged,
    )


def _model_offset(model_id: str, folds: int) -> int:
    digest = hashlib.blake2b(model_id.encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") % folds


def _balanced_folds(rows: list[_Observation], folds: int) -> dict[tuple[str, str], int]:
    families_by_model: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        families_by_model[row.model_id].add(row.family_id)
    assignment: dict[tuple[str, str], int] = {}
    for model_id, families in families_by_model.items():
        offset = _model_offset(model_id, folds)
        for index, family_id in enumerate(sorted(families)):
            assignment[(model_id, family_id)] = (offset + index) % folds
    return assignment


def cross_validate_estimator(
    models: list[ModelDefinition],
    benchmarks: list[BenchmarkDefinition],
    observations: list[BenchmarkObservation],
    adoption: list[BenchmarkAdoptionSnapshot] | None = None,
    *,
    config: EstimatorConfig | None = None,
    folds: int = 5,
    as_of: date | None = None,
) -> EstimatorCrossValidation:
    """Hold out whole model×benchmark-family groups and predict raw normalized scores."""
    if folds < 2:
        raise ValueError("folds must be at least 2")
    config = config or EstimatorConfig(jackknife_uncertainty=False)
    if config.jackknife_uncertainty:
        config = EstimatorConfig(**{**asdict(config), "jackknife_uncertainty": False})
    as_of = as_of or date.today()
    adoption = adoption or []
    rows, model_ids, benchmark_ids, _, _, _ = _prepare(
        models, benchmarks, observations, adoption, config, as_of
    )
    assignment = _balanced_folds(rows, folds)

    observed_all: list[float] = []
    predicted_all: list[float] = []
    baseline_all: list[float] = []
    weight_all: list[float] = []
    residual_observed: list[float] = []
    residual_predicted: list[float] = []

    for fold in range(folds):
        training = [
            row
            for row in rows
            if assignment[(row.model_id, row.family_id)] != fold
        ]
        validation = [
            row
            for row in rows
            if assignment[(row.model_id, row.family_id)] == fold
        ]
        if not training or not validation:
            continue
        state, _, _ = _fit_state(training, model_ids, benchmark_ids, config)
        by_benchmark_training: dict[str, list[_Observation]] = defaultdict(list)
        for row in training:
            by_benchmark_training[row.benchmark_id].append(row)

        for row in validation:
            if not by_benchmark_training.get(row.benchmark_id):
                continue
            prediction = config.score_unit_points * (
                state.intercept[row.benchmark_id]
                + state.loading[row.benchmark_id] * state.general[row.model_id]
            )
            baseline = config.score_unit_points * state.intercept[row.benchmark_id]
            observed_all.append(row.score_points)
            predicted_all.append(prediction)
            baseline_all.append(baseline)
            weight_all.append(row.weight)
            residual_observed.append(row.score_points - baseline)
            residual_predicted.append(prediction - baseline)

    if not observed_all:
        raise ValueError("Estimator cross-validation produced no held-out observations")
    observed = np.asarray(observed_all, dtype=float)
    predicted = np.asarray(predicted_all, dtype=float)
    baseline = np.asarray(baseline_all, dtype=float)
    weights = np.asarray(weight_all, dtype=float)
    model_sse = float(np.sum(weights * np.square(observed - predicted)))
    baseline_sse = float(np.sum(weights * np.square(observed - baseline)))
    model_rmse = math.sqrt(model_sse / max(float(weights.sum()), 1e-12))
    baseline_rmse = math.sqrt(baseline_sse / max(float(weights.sum()), 1e-12))
    improvement = 1.0 - model_rmse / baseline_rmse if baseline_rmse > 1e-12 else 0.0
    residual_r2 = 1.0 - model_sse / baseline_sse if baseline_sse > 1e-12 else 0.0
    rho = spearmanr(residual_observed, residual_predicted).statistic
    return EstimatorCrossValidation(
        folds=folds,
        heldout_observations=len(observed_all),
        model_rmse_points=round(model_rmse, 3),
        benchmark_only_rmse_points=round(baseline_rmse, 3),
        relative_rmse_improvement=round(improvement, 4),
        residual_r2=round(residual_r2, 4),
        residual_spearman=round(float(rho), 4) if np.isfinite(rho) else 0.0,
    )
