from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import date
from urllib.parse import urlparse

import numpy as np
import pandas as pd

from .benchmark_quality import BenchmarkQualityRow, BenchmarkTier, rank_benchmarks
from .factors import _filter_matrix, _normalized_matrix
from .novelty import ExposureTier, classify_exposure
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
class HierarchicalConfig:
    minimum_models_per_benchmark: int = 5
    minimum_benchmarks_per_model: int = 5
    minimum_families_per_model: int = 5
    maximum_iterations: int = 250
    tolerance: float = 1e-6
    ridge_general: float = 0.35
    ridge_domain: float = 3.0
    ridge_benchmark_loading: float = 1.5
    ridge_harness: float = 4.0
    ridge_ecosystem: float = 6.0
    ridge_global_novelty: float = 2.0
    ridge_model_novelty: float = 5.0
    minimum_ecosystem_observations: int = 2
    minimum_ecosystem_models: int = 3
    benchmark_loading_minimum: float = 0.05
    benchmark_loading_maximum: float = 2.5


@dataclass(frozen=True)
class HierarchicalModelEstimate:
    model_id: str
    general_z: float
    ci_low: float
    ci_high: float
    benchmark_count: int
    family_count: int
    coverage: float
    effective_evidence: float
    novelty_deviation: float
    domain_residuals: dict[str, float]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class HierarchicalBenchmarkEstimate:
    benchmark_id: str
    family_id: str
    tier: str
    evidence_weight: float
    general_loading: float
    observed_models: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class HarnessEffect:
    benchmark_id: str
    harness: str
    effect_z: float
    observations: int


@dataclass(frozen=True)
class EcosystemEffect:
    model_id: str
    ecosystem: str
    effect_z: float
    observations: int


@dataclass(frozen=True)
class HierarchicalResult:
    models: list[HierarchicalModelEstimate]
    benchmarks: list[HierarchicalBenchmarkEstimate]
    harness_effects: list[HarnessEffect]
    ecosystem_effects: list[EcosystemEffect]
    retained_models: list[str]
    retained_benchmarks: list[str]
    observed_cells: int
    possible_cells: int
    density: float
    weighted_rmse: float
    weighted_r2: float
    general_only_r2: float
    global_novelty_effect: float
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
                "weighted_rmse": self.weighted_rmse,
                "weighted_r2": self.weighted_r2,
                "general_only_r2": self.general_only_r2,
                "global_novelty_effect": self.global_novelty_effect,
                "iterations": self.iterations,
                "converged": self.converged,
            },
            "models": [row.to_dict() for row in self.models],
            "benchmarks": [row.to_dict() for row in self.benchmarks],
            "harness_effects": [asdict(row) for row in self.harness_effects],
            "ecosystem_effects": [asdict(row) for row in self.ecosystem_effects],
        }


@dataclass
class _PreparedObservation:
    model_id: str
    benchmark_id: str
    z: float
    weight: float
    family_id: str
    harness: str | None
    ecosystem: str | None
    novelty_signal: float
    exposure_known: bool


@dataclass
class _FitState:
    general: dict[str, float] = field(default_factory=dict)
    domain: dict[str, np.ndarray] = field(default_factory=dict)
    novelty: dict[str, float] = field(default_factory=dict)
    loading: dict[str, float] = field(default_factory=dict)
    harness: dict[tuple[str, str], float] = field(default_factory=dict)
    ecosystem: dict[tuple[str, str], float] = field(default_factory=dict)
    global_novelty: float = 0.0


def _root_domain(url: str | None) -> str | None:
    if not url:
        return None
    host = (urlparse(url).hostname or "").lower().removeprefix("www.")
    if not host:
        return None
    parts = host.split(".")
    if len(parts) <= 2:
        return host
    return ".".join(parts[-2:])


def _novelty_signal(
    model: ModelDefinition,
    benchmark: BenchmarkDefinition,
    observation: BenchmarkObservation,
) -> tuple[float, bool]:
    assessment = classify_exposure(
        model,
        benchmark,
        revision_at=observation.model_revision_at,
    )
    if assessment.tier in {
        ExposureTier.GUARANTEED_UNSEEN_POST_RELEASE,
        ExposureTier.DISCLOSED_CUTOFF_UNSEEN,
    }:
        return 1.0, True
    if assessment.tier in {
        ExposureTier.SEALED_TEST,
        ExposureTier.ROTATING_LOW_EXPOSURE,
    }:
        return 0.8, True
    if assessment.tier == ExposureTier.LIKELY_UNSEEN_SHORT_LEAD:
        return 0.5, True
    if assessment.tier == ExposureTier.EXPOSURE_POSSIBLE:
        return 0.0, True
    return 0.0, False


def _quality_map(
    benchmarks: list[BenchmarkDefinition],
    models: list[ModelDefinition],
    observations: list[BenchmarkObservation],
    adoption: list[BenchmarkAdoptionSnapshot],
    as_of: date,
) -> dict[str, BenchmarkQualityRow]:
    return {
        row.benchmark_id: row
        for row in rank_benchmarks(
            benchmarks,
            models,
            observations,
            adoption,
            as_of=as_of,
        )
    }


def _prepare(
    models: list[ModelDefinition],
    benchmarks: list[BenchmarkDefinition],
    observations: list[BenchmarkObservation],
    adoption: list[BenchmarkAdoptionSnapshot],
    config: HierarchicalConfig,
    as_of: date,
) -> tuple[
    list[_PreparedObservation],
    pd.DataFrame,
    dict[str, BenchmarkDefinition],
    dict[str, ModelDefinition],
    dict[str, BenchmarkQualityRow],
]:
    model_by_id = {row.id: row for row in models}
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
        raise ValueError("Insufficient overlapping data for hierarchical estimation")

    stds = frame.std(axis=0, skipna=True, ddof=0)
    keep = list(stds[stds > 1e-8].index)
    frame = frame.loc[:, keep]
    if frame.empty:
        raise ValueError("No retained benchmark has non-zero frontier discrimination")
    means = frame.mean(axis=0, skipna=True)
    stds = frame.std(axis=0, skipna=True, ddof=0)
    retained_models = set(str(item) for item in frame.index)
    retained_benchmarks = set(str(item) for item in frame.columns)
    quality = _quality_map(benchmarks, models, observations, adoption, as_of)

    rows: list[_PreparedObservation] = []
    for observation in observations:
        if (
            observation.model_id not in retained_models
            or observation.benchmark_id not in retained_benchmarks
        ):
            continue
        benchmark = benchmark_by_id.get(observation.benchmark_id)
        model = model_by_id.get(observation.model_id)
        quality_row = quality.get(observation.benchmark_id)
        if benchmark is None or model is None or quality_row is None:
            continue
        score = normalize_score(observation.score, benchmark)
        z = (score - float(means[benchmark.id])) / float(stds[benchmark.id])
        benchmark_weight = quality_row.family_adjusted_weight * _TIER_WEIGHT[quality_row.tier]
        source_weight = SOURCE_GRADE_WEIGHT[observation.source_grade]
        novelty_signal, exposure_known = _novelty_signal(model, benchmark, observation)
        rows.append(
            _PreparedObservation(
                model_id=model.id,
                benchmark_id=benchmark.id,
                z=float(z),
                weight=max(float(benchmark_weight * source_weight), 1e-4),
                family_id=benchmark.family_id or benchmark.id,
                harness=observation.harness.strip() if observation.harness else None,
                ecosystem=_root_domain(observation.source_url),
                novelty_signal=novelty_signal,
                exposure_known=exposure_known,
            )
        )
    if not rows:
        raise ValueError("No usable observations remained after filtering")
    mean_weight = float(np.mean([row.weight for row in rows]))
    for row in rows:
        row.weight /= mean_weight
    return rows, frame, benchmark_by_id, model_by_id, quality


def _centered_capability_loadings(
    benchmark_ids: list[str],
    benchmark_by_id: dict[str, BenchmarkDefinition],
    quality: dict[str, BenchmarkQualityRow],
) -> tuple[list[Capability], dict[str, np.ndarray]]:
    capabilities = list(Capability)
    raw: dict[str, np.ndarray] = {}
    weights = []
    vectors = []
    for benchmark_id in benchmark_ids:
        benchmark = benchmark_by_id[benchmark_id]
        vector = np.asarray(
            [benchmark.capability_loadings.get(capability, 0.0) for capability in capabilities],
            dtype=float,
        )
        raw[benchmark_id] = vector
        vectors.append(vector)
        weights.append(max(quality[benchmark_id].family_adjusted_weight, 1e-4))
    center = np.average(np.vstack(vectors), axis=0, weights=np.asarray(weights, dtype=float))
    return capabilities, {key: value - center for key, value in raw.items()}


def _active_harness_keys(rows: list[_PreparedObservation]) -> set[tuple[str, str]]:
    by_benchmark: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        if row.harness:
            by_benchmark[row.benchmark_id].add(row.harness)
    active_benchmarks = {
        benchmark_id for benchmark_id, names in by_benchmark.items() if len(names) >= 2
    }
    return {
        (row.benchmark_id, row.harness)
        for row in rows
        if row.harness and row.benchmark_id in active_benchmarks
    }


def _active_ecosystem_keys(
    rows: list[_PreparedObservation], config: HierarchicalConfig
) -> set[tuple[str, str]]:
    pair_count: Counter[tuple[str, str]] = Counter()
    models_by_ecosystem: dict[str, set[str]] = defaultdict(set)
    ecosystems_by_model: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        if not row.ecosystem:
            continue
        pair_count[(row.model_id, row.ecosystem)] += 1
        models_by_ecosystem[row.ecosystem].add(row.model_id)
        ecosystems_by_model[row.model_id].add(row.ecosystem)
    return {
        key
        for key, count in pair_count.items()
        if count >= config.minimum_ecosystem_observations
        and len(models_by_ecosystem[key[1]]) >= config.minimum_ecosystem_models
        and len(ecosystems_by_model[key[0]]) >= 2
    }


def _prediction(
    row: _PreparedObservation,
    state: _FitState,
    centered_loadings: dict[str, np.ndarray],
) -> float:
    value = state.loading[row.benchmark_id] * state.general[row.model_id]
    value += float(np.dot(centered_loadings[row.benchmark_id], state.domain[row.model_id]))
    if row.harness:
        value += state.harness.get((row.benchmark_id, row.harness), 0.0)
    if row.ecosystem:
        value += state.ecosystem.get((row.model_id, row.ecosystem), 0.0)
    if row.exposure_known:
        value += (
            state.global_novelty + state.novelty[row.model_id]
        ) * row.novelty_signal
    return value


def _weighted_loss(
    rows: list[_PreparedObservation],
    state: _FitState,
    centered_loadings: dict[str, np.ndarray],
) -> float:
    return float(
        sum(
            row.weight
            * (row.z - _prediction(row, state, centered_loadings)) ** 2
            for row in rows
        )
    )


def _update_models(
    rows_by_model: dict[str, list[_PreparedObservation]],
    state: _FitState,
    centered_loadings: dict[str, np.ndarray],
    config: HierarchicalConfig,
) -> None:
    capability_count = len(next(iter(centered_loadings.values())))
    penalty = np.diag(
        [config.ridge_general]
        + [config.ridge_domain] * capability_count
        + [config.ridge_model_novelty]
    )
    for model_id, rows in rows_by_model.items():
        design = []
        target = []
        weights = []
        for row in rows:
            nuisance = 0.0
            if row.harness:
                nuisance += state.harness.get((row.benchmark_id, row.harness), 0.0)
            if row.ecosystem:
                nuisance += state.ecosystem.get((model_id, row.ecosystem), 0.0)
            if row.exposure_known:
                nuisance += state.global_novelty * row.novelty_signal
            design.append(
                np.concatenate(
                    (
                        np.asarray([state.loading[row.benchmark_id]]),
                        centered_loadings[row.benchmark_id],
                        np.asarray([row.novelty_signal if row.exposure_known else 0.0]),
                    )
                )
            )
            target.append(row.z - nuisance)
            weights.append(row.weight)
        x = np.vstack(design)
        y = np.asarray(target, dtype=float)
        w = np.asarray(weights, dtype=float)
        xtw = x.T * w
        params = np.linalg.solve(xtw @ x + penalty, xtw @ y)
        state.general[model_id] = float(params[0])
        domain = params[1 : 1 + capability_count]
        domain -= float(domain.mean())
        state.domain[model_id] = domain
        state.novelty[model_id] = float(params[-1])


def _update_loadings(
    rows_by_benchmark: dict[str, list[_PreparedObservation]],
    state: _FitState,
    centered_loadings: dict[str, np.ndarray],
    config: HierarchicalConfig,
) -> None:
    for benchmark_id, rows in rows_by_benchmark.items():
        numerator = config.ridge_benchmark_loading
        denominator = config.ridge_benchmark_loading
        for row in rows:
            residual = row.z
            residual -= float(
                np.dot(centered_loadings[benchmark_id], state.domain[row.model_id])
            )
            if row.harness:
                residual -= state.harness.get((benchmark_id, row.harness), 0.0)
            if row.ecosystem:
                residual -= state.ecosystem.get((row.model_id, row.ecosystem), 0.0)
            if row.exposure_known:
                residual -= (
                    state.global_novelty + state.novelty[row.model_id]
                ) * row.novelty_signal
            general = state.general[row.model_id]
            numerator += row.weight * general * residual
            denominator += row.weight * general * general
        value = numerator / max(denominator, 1e-12)
        state.loading[benchmark_id] = float(
            np.clip(
                value,
                config.benchmark_loading_minimum,
                config.benchmark_loading_maximum,
            )
        )


def _update_global_novelty(
    rows: list[_PreparedObservation],
    state: _FitState,
    centered_loadings: dict[str, np.ndarray],
    config: HierarchicalConfig,
) -> None:
    numerator = 0.0
    denominator = config.ridge_global_novelty
    for row in rows:
        if not row.exposure_known or row.novelty_signal <= 0:
            continue
        residual = row.z
        residual -= state.loading[row.benchmark_id] * state.general[row.model_id]
        residual -= float(
            np.dot(centered_loadings[row.benchmark_id], state.domain[row.model_id])
        )
        residual -= state.novelty[row.model_id] * row.novelty_signal
        if row.harness:
            residual -= state.harness.get((row.benchmark_id, row.harness), 0.0)
        if row.ecosystem:
            residual -= state.ecosystem.get((row.model_id, row.ecosystem), 0.0)
        numerator += row.weight * row.novelty_signal * residual
        denominator += row.weight * row.novelty_signal * row.novelty_signal
    state.global_novelty = numerator / max(denominator, 1e-12)


def _update_harness(
    rows: list[_PreparedObservation],
    state: _FitState,
    centered_loadings: dict[str, np.ndarray],
    active_keys: set[tuple[str, str]],
    config: HierarchicalConfig,
) -> None:
    grouped: dict[tuple[str, str], list[_PreparedObservation]] = defaultdict(list)
    for row in rows:
        key = (row.benchmark_id, row.harness) if row.harness else None
        if key in active_keys:
            grouped[key].append(row)
    for key, group in grouped.items():
        numerator = 0.0
        denominator = config.ridge_harness
        for row in group:
            current = state.harness.get(key, 0.0)
            residual = row.z - (_prediction(row, state, centered_loadings) - current)
            numerator += row.weight * residual
            denominator += row.weight
        state.harness[key] = numerator / max(denominator, 1e-12)

    by_benchmark: dict[str, list[tuple[tuple[str, str], float]]] = defaultdict(list)
    for key, group in grouped.items():
        support = sum(row.weight for row in group)
        by_benchmark[key[0]].append((key, support))
    for entries in by_benchmark.values():
        total = sum(support for _, support in entries)
        if total <= 0:
            continue
        mean = sum(state.harness[key] * support for key, support in entries) / total
        for key, _ in entries:
            state.harness[key] -= mean


def _update_ecosystem(
    rows: list[_PreparedObservation],
    state: _FitState,
    centered_loadings: dict[str, np.ndarray],
    active_keys: set[tuple[str, str]],
    config: HierarchicalConfig,
) -> None:
    grouped: dict[tuple[str, str], list[_PreparedObservation]] = defaultdict(list)
    for row in rows:
        key = (row.model_id, row.ecosystem) if row.ecosystem else None
        if key in active_keys:
            grouped[key].append(row)
    for key, group in grouped.items():
        numerator = 0.0
        denominator = config.ridge_ecosystem
        for row in group:
            current = state.ecosystem.get(key, 0.0)
            residual = row.z - (_prediction(row, state, centered_loadings) - current)
            numerator += row.weight * residual
            denominator += row.weight
        state.ecosystem[key] = numerator / max(denominator, 1e-12)

    by_model: dict[str, list[tuple[tuple[str, str], float]]] = defaultdict(list)
    for key, group in grouped.items():
        support = sum(row.weight for row in group)
        by_model[key[0]].append((key, support))
    for entries in by_model.values():
        total = sum(support for _, support in entries)
        if total <= 0:
            continue
        mean = sum(state.ecosystem[key] * support for key, support in entries) / total
        for key, _ in entries:
            state.ecosystem[key] -= mean


def _effective_evidence(rows: list[_PreparedObservation]) -> float:
    by_family: dict[str, float] = defaultdict(float)
    for row in rows:
        by_family[row.family_id] += row.weight
    values = np.asarray(list(by_family.values()), dtype=float)
    denominator = float(np.square(values).sum())
    if denominator <= 0:
        return 0.0
    return float(values.sum() ** 2 / denominator)


def _model_uncertainty(
    rows: list[_PreparedObservation],
    state: _FitState,
    centered_loadings: dict[str, np.ndarray],
    config: HierarchicalConfig,
    residual_variance: float,
    general_scale: float,
) -> float:
    capability_count = len(next(iter(centered_loadings.values())))
    design = []
    weights = []
    for row in rows:
        design.append(
            np.concatenate(
                (
                    np.asarray([state.loading[row.benchmark_id]]),
                    centered_loadings[row.benchmark_id],
                    np.asarray([row.novelty_signal if row.exposure_known else 0.0]),
                )
            )
        )
        weights.append(row.weight)
    x = np.vstack(design)
    w = np.asarray(weights, dtype=float)
    penalty = np.diag(
        [config.ridge_general]
        + [config.ridge_domain] * capability_count
        + [config.ridge_model_novelty]
    )
    precision = (x.T * w) @ x + penalty
    covariance = residual_variance * np.linalg.inv(precision)
    raw_se = math.sqrt(max(float(covariance[0, 0]), 0.0))
    return raw_se / max(general_scale, 1e-12)


def fit_hierarchical(
    models: list[ModelDefinition],
    benchmarks: list[BenchmarkDefinition],
    observations: list[BenchmarkObservation],
    adoption: list[BenchmarkAdoptionSnapshot] | None = None,
    *,
    config: HierarchicalConfig | None = None,
    as_of: date | None = None,
) -> HierarchicalResult:
    """Fit a quality-weighted hierarchical model to the sparse benchmark matrix.

    The estimator separates a learned general factor from regularized capability
    deviations. Hand-authored benchmark capability loadings act as a structural prior,
    not as the final discovered taxonomy. Within-benchmark harness effects, model×source
    ecosystem deviations and model-specific novelty sensitivity are shrinkage terms.
    """
    config = config or HierarchicalConfig()
    as_of = as_of or date.today()
    adoption = adoption or []
    rows, frame, benchmark_by_id, _, quality = _prepare(
        models,
        benchmarks,
        observations,
        adoption,
        config,
        as_of,
    )
    model_ids = [str(item) for item in frame.index]
    benchmark_ids = [str(item) for item in frame.columns]
    capabilities, centered_loadings = _centered_capability_loadings(
        benchmark_ids,
        benchmark_by_id,
        quality,
    )
    rows_by_model: dict[str, list[_PreparedObservation]] = defaultdict(list)
    rows_by_benchmark: dict[str, list[_PreparedObservation]] = defaultdict(list)
    for row in rows:
        rows_by_model[row.model_id].append(row)
        rows_by_benchmark[row.benchmark_id].append(row)

    state = _FitState(
        general={model_id: 0.0 for model_id in model_ids},
        domain={model_id: np.zeros(len(capabilities), dtype=float) for model_id in model_ids},
        novelty={model_id: 0.0 for model_id in model_ids},
        loading={benchmark_id: 1.0 for benchmark_id in benchmark_ids},
    )
    for model_id, model_rows in rows_by_model.items():
        state.general[model_id] = float(
            np.average(
                np.asarray([row.z for row in model_rows]),
                weights=np.asarray([row.weight for row in model_rows]),
            )
        )

    active_harness = _active_harness_keys(rows)
    active_ecosystem = _active_ecosystem_keys(rows, config)
    state.harness = {key: 0.0 for key in active_harness}
    state.ecosystem = {key: 0.0 for key in active_ecosystem}

    previous = float("inf")
    converged = False
    iteration = 0
    for iteration in range(1, config.maximum_iterations + 1):
        _update_models(rows_by_model, state, centered_loadings, config)
        _update_loadings(rows_by_benchmark, state, centered_loadings, config)
        _update_global_novelty(rows, state, centered_loadings, config)
        _update_harness(rows, state, centered_loadings, active_harness, config)
        _update_ecosystem(rows, state, centered_loadings, active_ecosystem, config)
        loss = _weighted_loss(rows, state, centered_loadings)
        if previous < float("inf"):
            relative = abs(previous - loss) / max(previous, 1e-12)
            if relative < config.tolerance:
                converged = True
                break
        previous = loss

    predictions = np.asarray(
        [_prediction(row, state, centered_loadings) for row in rows], dtype=float
    )
    observed = np.asarray([row.z for row in rows], dtype=float)
    weights = np.asarray([row.weight for row in rows], dtype=float)
    residual = observed - predictions
    weighted_sse = float(np.sum(weights * np.square(residual)))
    weighted_mean = float(np.average(observed, weights=weights))
    weighted_sst = float(np.sum(weights * np.square(observed - weighted_mean)))
    weighted_r2 = 1.0 - weighted_sse / weighted_sst if weighted_sst > 0 else 0.0
    weighted_rmse = math.sqrt(weighted_sse / max(float(weights.sum()), 1e-12))

    general_prediction = np.asarray(
        [state.loading[row.benchmark_id] * state.general[row.model_id] for row in rows],
        dtype=float,
    )
    general_sse = float(np.sum(weights * np.square(observed - general_prediction)))
    general_only_r2 = 1.0 - general_sse / weighted_sst if weighted_sst > 0 else 0.0

    raw_general = np.asarray([state.general[model_id] for model_id in model_ids], dtype=float)
    general_mean = float(raw_general.mean())
    general_scale = float(raw_general.std(ddof=0))
    if general_scale <= 1e-8:
        raise ValueError("Hierarchical general factor collapsed to zero variance")
    residual_variance = weighted_sse / max(float(weights.sum()), 1.0)
    total_families = len({benchmark_by_id[item].family_id or item for item in benchmark_ids})

    model_estimates: list[HierarchicalModelEstimate] = []
    for model_id in model_ids:
        model_rows = rows_by_model[model_id]
        general_z = (state.general[model_id] - general_mean) / general_scale
        se = _model_uncertainty(
            model_rows,
            state,
            centered_loadings,
            config,
            residual_variance,
            general_scale,
        )
        families = {row.family_id for row in model_rows}
        domains = {
            capability.value: float(state.domain[model_id][index])
            for index, capability in enumerate(capabilities)
        }
        model_estimates.append(
            HierarchicalModelEstimate(
                model_id=model_id,
                general_z=round(float(general_z), 4),
                ci_low=round(float(general_z - 1.96 * se), 4),
                ci_high=round(float(general_z + 1.96 * se), 4),
                benchmark_count=len({row.benchmark_id for row in model_rows}),
                family_count=len(families),
                coverage=round(len(families) / max(total_families, 1), 4),
                effective_evidence=round(_effective_evidence(model_rows), 3),
                novelty_deviation=round(float(state.novelty[model_id]), 4),
                domain_residuals={key: round(value, 4) for key, value in domains.items()},
            )
        )
    model_estimates.sort(key=lambda row: row.general_z, reverse=True)

    benchmark_estimates = [
        HierarchicalBenchmarkEstimate(
            benchmark_id=benchmark_id,
            family_id=benchmark_by_id[benchmark_id].family_id or benchmark_id,
            tier=quality[benchmark_id].tier.value,
            evidence_weight=round(quality[benchmark_id].family_adjusted_weight, 4),
            general_loading=round(float(state.loading[benchmark_id] * general_scale), 4),
            observed_models=len({row.model_id for row in rows_by_benchmark[benchmark_id]}),
        )
        for benchmark_id in benchmark_ids
    ]
    benchmark_estimates.sort(key=lambda row: abs(row.general_loading), reverse=True)

    harness_effects = [
        HarnessEffect(
            benchmark_id=key[0],
            harness=key[1],
            effect_z=round(float(value), 4),
            observations=sum(
                1
                for row in rows
                if row.benchmark_id == key[0] and row.harness == key[1]
            ),
        )
        for key, value in state.harness.items()
    ]
    harness_effects.sort(key=lambda row: abs(row.effect_z), reverse=True)

    ecosystem_effects = [
        EcosystemEffect(
            model_id=key[0],
            ecosystem=key[1],
            effect_z=round(float(value), 4),
            observations=sum(
                1
                for row in rows
                if row.model_id == key[0] and row.ecosystem == key[1]
            ),
        )
        for key, value in state.ecosystem.items()
    ]
    ecosystem_effects.sort(key=lambda row: abs(row.effect_z), reverse=True)

    observed_cells = int(frame.notna().sum().sum())
    possible_cells = int(frame.shape[0] * frame.shape[1])
    return HierarchicalResult(
        models=model_estimates,
        benchmarks=benchmark_estimates,
        harness_effects=harness_effects,
        ecosystem_effects=ecosystem_effects,
        retained_models=model_ids,
        retained_benchmarks=benchmark_ids,
        observed_cells=observed_cells,
        possible_cells=possible_cells,
        density=round(observed_cells / possible_cells, 4),
        weighted_rmse=round(weighted_rmse, 4),
        weighted_r2=round(weighted_r2, 4),
        general_only_r2=round(general_only_r2, 4),
        global_novelty_effect=round(float(state.global_novelty), 4),
        iterations=iteration,
        converged=converged,
    )
