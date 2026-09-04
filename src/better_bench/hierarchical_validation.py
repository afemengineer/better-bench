from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import date

import numpy as np

from .hierarchical import (
    HierarchicalConfig,
    _FitState,
    _PreparedObservation,
    _active_ecosystem_keys,
    _active_harness_keys,
    _centered_capability_loadings,
    _prediction,
    _prepare,
    _update_ecosystem,
    _update_global_novelty,
    _update_harness,
    _update_loadings,
    _update_models,
    _weighted_loss,
)
from .schema import (
    BenchmarkAdoptionSnapshot,
    BenchmarkDefinition,
    BenchmarkObservation,
    ModelDefinition,
)
from .scoring import normalize_score


@dataclass(frozen=True)
class CrossValidationModelDiagnostic:
    model_id: str
    heldout_observations: int
    full_rmse: float
    general_only_rmse: float
    full_bias: float


@dataclass(frozen=True)
class CrossValidationResult:
    folds: int
    heldout_observations: int
    full_rmse: float
    general_only_rmse: float
    full_r2: float
    general_only_r2: float
    relative_rmse_improvement: float
    model_diagnostics: list[CrossValidationModelDiagnostic]


def _fold_index(model_id: str, family_id: str, folds: int) -> int:
    digest = hashlib.blake2b(
        f"{model_id}|{family_id}".encode("utf-8"), digest_size=8
    ).digest()
    return int.from_bytes(digest, "big") % folds


def _fit_state(
    rows: list[_PreparedObservation],
    model_ids: list[str],
    benchmark_ids: list[str],
    centered_loadings: dict[str, np.ndarray],
    config: HierarchicalConfig,
) -> _FitState:
    capability_count = len(next(iter(centered_loadings.values())))
    rows_by_model: dict[str, list[_PreparedObservation]] = defaultdict(list)
    rows_by_benchmark: dict[str, list[_PreparedObservation]] = defaultdict(list)
    for row in rows:
        rows_by_model[row.model_id].append(row)
        rows_by_benchmark[row.benchmark_id].append(row)

    state = _FitState(
        general={model_id: 0.0 for model_id in model_ids},
        domain={
            model_id: np.zeros(capability_count, dtype=float) for model_id in model_ids
        },
        novelty={model_id: 0.0 for model_id in model_ids},
        loading={benchmark_id: 1.0 for benchmark_id in benchmark_ids},
    )
    for model_id, model_rows in rows_by_model.items():
        if model_rows:
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
    for _ in range(config.maximum_iterations):
        _update_models(rows_by_model, state, centered_loadings, config)
        _update_loadings(rows_by_benchmark, state, centered_loadings, config)
        _update_global_novelty(rows, state, centered_loadings, config)
        _update_harness(rows, state, centered_loadings, active_harness, config)
        _update_ecosystem(rows, state, centered_loadings, active_ecosystem, config)
        loss = _weighted_loss(rows, state, centered_loadings)
        if previous < float("inf"):
            relative = abs(previous - loss) / max(previous, 1e-12)
            if relative < config.tolerance:
                break
        previous = loss
    return state


def _general_only_config(config: HierarchicalConfig) -> HierarchicalConfig:
    huge = 1e9
    return replace(
        config,
        ridge_domain=huge,
        ridge_harness=huge,
        ridge_ecosystem=huge,
        ridge_global_novelty=huge,
        ridge_model_novelty=huge,
    )


def _raw_normalized_scores(
    observations: list[BenchmarkObservation],
    benchmark_by_id: dict[str, BenchmarkDefinition],
    retained_pairs: set[tuple[str, str]],
) -> dict[tuple[str, str], float]:
    values: dict[tuple[str, str], list[float]] = defaultdict(list)
    for observation in observations:
        key = (observation.model_id, observation.benchmark_id)
        benchmark = benchmark_by_id.get(observation.benchmark_id)
        if key not in retained_pairs or benchmark is None:
            continue
        values[key].append(normalize_score(observation.score, benchmark))
    return {key: float(np.mean(scores)) for key, scores in values.items()}


def _training_standardization(
    rows: list[_PreparedObservation],
    raw_scores: dict[tuple[str, str], float],
) -> dict[str, tuple[float, float]]:
    by_benchmark: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        key = (row.model_id, row.benchmark_id)
        if key in raw_scores:
            by_benchmark[row.benchmark_id].append(raw_scores[key])

    result: dict[str, tuple[float, float]] = {}
    for benchmark_id, scores in by_benchmark.items():
        if len(scores) < 2:
            continue
        values = np.asarray(scores, dtype=float)
        std = float(values.std(ddof=0))
        if std <= 1e-8:
            continue
        result[benchmark_id] = (float(values.mean()), std)
    return result


def _standardize_rows(
    rows: list[_PreparedObservation],
    raw_scores: dict[tuple[str, str], float],
    stats: dict[str, tuple[float, float]],
) -> list[_PreparedObservation]:
    standardized: list[_PreparedObservation] = []
    for row in rows:
        benchmark_stats = stats.get(row.benchmark_id)
        raw = raw_scores.get((row.model_id, row.benchmark_id))
        if benchmark_stats is None or raw is None:
            continue
        mean, std = benchmark_stats
        standardized.append(replace(row, z=(raw - mean) / std))
    return standardized


def _weighted_metrics(
    observed: np.ndarray,
    predicted: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, float]:
    sse = float(np.sum(weights * np.square(observed - predicted)))
    baseline = float(np.sum(weights * np.square(observed)))
    rmse = math.sqrt(sse / max(float(weights.sum()), 1e-12))
    r2 = 1.0 - sse / baseline if baseline > 1e-12 else 0.0
    return rmse, r2


def cross_validate_hierarchical(
    models: list[ModelDefinition],
    benchmarks: list[BenchmarkDefinition],
    observations: list[BenchmarkObservation],
    adoption: list[BenchmarkAdoptionSnapshot] | None = None,
    *,
    config: HierarchicalConfig | None = None,
    folds: int = 5,
    as_of: date | None = None,
) -> CrossValidationResult:
    """Cross-validate by withholding model×benchmark-family groups.

    Each model-family group is assigned wholly to one deterministic fold, preventing
    protocol variants from the same benchmark family from leaking into training for that
    model. Benchmark centering/scaling is recomputed from training models only in every
    fold. Benchmark-quality weights remain frozen from the full metadata snapshot because
    they are measurement priors, not target values.
    """
    if folds < 2:
        raise ValueError("folds must be at least 2")
    config = config or HierarchicalConfig()
    as_of = as_of or date.today()
    adoption = adoption or []

    prepared, frame, benchmark_by_id, _, quality = _prepare(
        models,
        benchmarks,
        observations,
        adoption,
        config,
        as_of,
    )
    model_ids = [str(item) for item in frame.index]
    benchmark_ids = [str(item) for item in frame.columns]
    _, centered_loadings = _centered_capability_loadings(
        benchmark_ids,
        benchmark_by_id,
        quality,
    )
    retained_pairs = {(row.model_id, row.benchmark_id) for row in prepared}
    raw_scores = _raw_normalized_scores(
        observations,
        benchmark_by_id,
        retained_pairs,
    )

    fold_by_group = {
        (row.model_id, row.family_id): _fold_index(row.model_id, row.family_id, folds)
        for row in prepared
    }

    all_observed: list[float] = []
    all_full: list[float] = []
    all_general: list[float] = []
    all_weights: list[float] = []
    heldout_by_model: dict[str, list[tuple[float, float, float, float]]] = defaultdict(list)

    general_config = _general_only_config(config)
    for fold in range(folds):
        training_base = [
            row
            for row in prepared
            if fold_by_group[(row.model_id, row.family_id)] != fold
        ]
        validation_base = [
            row
            for row in prepared
            if fold_by_group[(row.model_id, row.family_id)] == fold
        ]
        stats = _training_standardization(training_base, raw_scores)
        training = _standardize_rows(training_base, raw_scores, stats)
        validation = _standardize_rows(validation_base, raw_scores, stats)
        if not training or not validation:
            continue

        full_state = _fit_state(
            training,
            model_ids,
            benchmark_ids,
            centered_loadings,
            config,
        )
        general_state = _fit_state(
            training,
            model_ids,
            benchmark_ids,
            centered_loadings,
            general_config,
        )

        for row in validation:
            full_prediction = _prediction(row, full_state, centered_loadings)
            general_prediction = _prediction(row, general_state, centered_loadings)
            all_observed.append(row.z)
            all_full.append(full_prediction)
            all_general.append(general_prediction)
            all_weights.append(row.weight)
            heldout_by_model[row.model_id].append(
                (row.z, full_prediction, general_prediction, row.weight)
            )

    if not all_observed:
        raise ValueError("Cross-validation produced no held-out observations")

    observed_array = np.asarray(all_observed, dtype=float)
    full_array = np.asarray(all_full, dtype=float)
    general_array = np.asarray(all_general, dtype=float)
    weight_array = np.asarray(all_weights, dtype=float)
    full_rmse, full_r2 = _weighted_metrics(observed_array, full_array, weight_array)
    general_rmse, general_r2 = _weighted_metrics(
        observed_array,
        general_array,
        weight_array,
    )

    model_diagnostics: list[CrossValidationModelDiagnostic] = []
    for model_id, values in heldout_by_model.items():
        observed_model = np.asarray([item[0] for item in values], dtype=float)
        full_model = np.asarray([item[1] for item in values], dtype=float)
        general_model = np.asarray([item[2] for item in values], dtype=float)
        weight_model = np.asarray([item[3] for item in values], dtype=float)
        full_model_rmse, _ = _weighted_metrics(
            observed_model,
            full_model,
            weight_model,
        )
        general_model_rmse, _ = _weighted_metrics(
            observed_model,
            general_model,
            weight_model,
        )
        bias = float(
            np.average(observed_model - full_model, weights=weight_model)
        )
        model_diagnostics.append(
            CrossValidationModelDiagnostic(
                model_id=model_id,
                heldout_observations=len(values),
                full_rmse=round(full_model_rmse, 4),
                general_only_rmse=round(general_model_rmse, 4),
                full_bias=round(bias, 4),
            )
        )
    model_diagnostics.sort(key=lambda row: row.full_rmse, reverse=True)

    improvement = 1.0 - full_rmse / general_rmse if general_rmse > 1e-12 else 0.0
    return CrossValidationResult(
        folds=folds,
        heldout_observations=len(all_observed),
        full_rmse=round(full_rmse, 4),
        general_only_rmse=round(general_rmse, 4),
        full_r2=round(full_r2, 4),
        general_only_r2=round(general_r2, 4),
        relative_rmse_improvement=round(improvement, 4),
        model_diagnostics=model_diagnostics,
    )
