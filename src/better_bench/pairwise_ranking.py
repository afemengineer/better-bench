from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PairwiseRankingConfig:
    """Configuration for coverage-neutral model ranking.

    ``minimum_shared_families`` is deliberately expressed in independent benchmark
    families, not raw rows. ``ridge`` controls shrinkage strength for weakly connected
    models. ``prior_mix`` controls how much of that shrinkage target comes from the
    observed-portfolio fixed estimator: 0 means population-mean shrinkage, 1 means the
    previous fixed-BBI prior. Keeping these separate prevents missing benchmark cells
    from silently re-entering the ranking through regularization.
    """

    minimum_shared_families: int = 2
    ridge: float = 1.0
    prior_mix: float = 1.0
    information_scale_points: float = 10.0


@dataclass(frozen=True)
class PairwiseEdge:
    left_model: str
    right_model: str
    delta_z: float
    weight: float
    shared_families: int
    shared_benchmarks: int


@dataclass(frozen=True)
class PairwiseRankingResult:
    scores: dict[str, float]
    prior_scores: dict[str, float]
    edges: list[PairwiseEdge]
    graph_information: dict[str, float]
    graph_standard_error: dict[str, float]
    score_scale: float


def _standardized_prior(state: Any, model_ids: list[str]) -> tuple[np.ndarray, float]:
    values = np.asarray([state.general[model_id] for model_id in model_ids], dtype=float)
    mean = float(values.mean())
    scale = float(values.std(ddof=0))
    if scale <= 1e-12:
        return np.zeros_like(values), 1.0
    return (values - mean) / scale, scale


def _aggregate_model_benchmark_rows(
    rows: Iterable[Any],
) -> dict[tuple[str, str], tuple[float, float, str]]:
    """Collapse repeated model×benchmark observations before constructing comparisons."""
    grouped: dict[tuple[str, str], list[Any]] = defaultdict(list)
    for row in rows:
        grouped[(row.model_id, row.benchmark_id)].append(row)

    result: dict[tuple[str, str], tuple[float, float, str]] = {}
    for key, values in grouped.items():
        weights = np.asarray([float(row.weight) for row in values], dtype=float)
        scores = np.asarray([float(row.score_points) for row in values], dtype=float)
        total = float(weights.sum())
        if total <= 1e-12:
            continue
        families = {str(row.family_id) for row in values}
        if len(families) != 1:
            raise ValueError(f"Benchmark {key[1]} maps to multiple families")
        result[key] = (
            float(np.average(scores, weights=weights)),
            total,
            next(iter(families)),
        )
    return result


def build_pairwise_edges(
    rows: Iterable[Any],
    model_ids: list[str],
    state: Any,
    *,
    score_unit_points: float,
    config: PairwiseRankingConfig,
) -> list[PairwiseEdge]:
    """Profile benchmark intercepts out of the fixed weighted least-squares model.

    For benchmark j with observation weights w_ij, weighted least squares obeys

        sum_i w_ij (x_ij - mean_w(x_j))^2
        = (1 / W_j) sum_{i<k} w_ij w_kj (x_ij - x_kj)^2.

    Therefore the exact pair information after eliminating benchmark difficulty is
    ``w_i * w_k / W`` rather than a heuristic such as ``sqrt(w_i*w_k)``. This is
    important for Better Bench because the heuristic gives a benchmark O(n^2)
    influence as more models are evaluated on it. The profiled construction preserves
    the intended benchmark/provenance information budget while still using only shared
    evidence. Learned benchmark discrimination supplies the conversion from score-point
    differences to standardized latent-general-intelligence differences.

    Family-adjusted benchmark weights are already embedded in each row by the fixed
    estimator's preparation layer, so protocol variants from one family share that
    family's budget. We still require a minimum number of distinct families before an
    edge is admitted.
    """
    if config.minimum_shared_families < 1:
        raise ValueError("minimum_shared_families must be at least 1")
    if config.information_scale_points <= 0:
        raise ValueError("information_scale_points must be positive")

    _, general_scale = _standardized_prior(state, model_ids)
    calibration: dict[str, tuple[float, float]] = {}
    for benchmark_id, loading in state.loading.items():
        loading_points_per_z = float(score_unit_points * loading * general_scale)
        if loading_points_per_z <= 1e-8:
            continue
        information_multiplier = (
            loading_points_per_z / config.information_scale_points
        ) ** 2
        calibration[str(benchmark_id)] = (
            loading_points_per_z,
            information_multiplier,
        )

    cells = _aggregate_model_benchmark_rows(rows)
    benchmark_total_weight: dict[str, float] = defaultdict(float)
    benchmarks_by_model: dict[str, set[str]] = defaultdict(set)
    for (model_id, benchmark_id), (_, weight, _) in cells.items():
        if benchmark_id not in calibration:
            continue
        benchmark_total_weight[benchmark_id] += weight
        benchmarks_by_model[model_id].add(benchmark_id)

    edges: list[PairwiseEdge] = []
    for left_index, left_model in enumerate(model_ids):
        for right_model in model_ids[left_index + 1 :]:
            shared = benchmarks_by_model[left_model] & benchmarks_by_model[right_model]
            family_values: dict[str, list[tuple[float, float]]] = defaultdict(list)
            for benchmark_id in shared:
                left_score, left_weight, family_id = cells[(left_model, benchmark_id)]
                right_score, right_weight, right_family_id = cells[
                    (right_model, benchmark_id)
                ]
                if family_id != right_family_id:
                    raise ValueError(
                        f"Benchmark {benchmark_id} has inconsistent family metadata"
                    )
                total_benchmark_weight = benchmark_total_weight[benchmark_id]
                if total_benchmark_weight <= 1e-12:
                    continue
                scale, information_multiplier = calibration[benchmark_id]
                delta_z = (left_score - right_score) / scale
                pair_information = (
                    left_weight
                    * right_weight
                    / total_benchmark_weight
                    * information_multiplier
                )
                if pair_information <= 1e-12:
                    continue
                family_values[family_id].append((delta_z, pair_information))

            if len(family_values) < config.minimum_shared_families:
                continue

            family_deltas: list[float] = []
            family_information: list[float] = []
            benchmark_count = 0
            for values in family_values.values():
                benchmark_count += len(values)
                information = np.asarray([item[1] for item in values], dtype=float)
                deltas = np.asarray([item[0] for item in values], dtype=float)
                total_information = float(information.sum())
                if total_information <= 1e-12:
                    continue
                family_deltas.append(float(np.average(deltas, weights=information)))
                family_information.append(total_information)

            if len(family_deltas) < config.minimum_shared_families:
                continue
            family_weights = np.asarray(family_information, dtype=float)
            family_delta_array = np.asarray(family_deltas, dtype=float)
            edge_weight = float(family_weights.sum())
            if edge_weight <= 1e-12:
                continue
            edges.append(
                PairwiseEdge(
                    left_model=left_model,
                    right_model=right_model,
                    delta_z=float(np.average(family_delta_array, weights=family_weights)),
                    weight=edge_weight,
                    shared_families=len(family_deltas),
                    shared_benchmarks=benchmark_count,
                )
            )
    return edges


def fit_pairwise_ranking(
    rows: Iterable[Any],
    model_ids: list[str],
    state: Any,
    *,
    score_unit_points: float,
    config: PairwiseRankingConfig | None = None,
) -> PairwiseRankingResult:
    """Fit a globally coherent ranking from comparable model pairs.

    Missing benchmark cells never appear as positive evidence. They only reduce graph
    connectivity. Weakly connected models shrink toward a target controlled explicitly
    by ``prior_mix``: population mean at 0, observed-portfolio fixed prior at 1.
    """
    config = config or PairwiseRankingConfig()
    if config.ridge <= 0:
        raise ValueError("ridge must be positive")
    if not 0.0 <= config.prior_mix <= 1.0:
        raise ValueError("prior_mix must be between 0 and 1")
    if not model_ids:
        raise ValueError("model_ids cannot be empty")

    prior, _ = _standardized_prior(state, model_ids)
    edges = build_pairwise_edges(
        rows,
        model_ids,
        state,
        score_unit_points=score_unit_points,
        config=config,
    )
    index = {model_id: position for position, model_id in enumerate(model_ids)}
    size = len(model_ids)
    laplacian = np.zeros((size, size), dtype=float)
    rhs = np.zeros(size, dtype=float)
    for edge in edges:
        left = index[edge.left_model]
        right = index[edge.right_model]
        weight = edge.weight
        laplacian[left, left] += weight
        laplacian[right, right] += weight
        laplacian[left, right] -= weight
        laplacian[right, left] -= weight
        rhs[left] += weight * edge.delta_z
        rhs[right] -= weight * edge.delta_z

    system = laplacian + config.ridge * np.eye(size)
    rhs += config.ridge * config.prior_mix * prior
    raw_scores = np.linalg.solve(system, rhs)
    raw_scores -= float(raw_scores.mean())
    score_scale = float(raw_scores.std(ddof=0))
    if score_scale <= 1e-12:
        score_scale = 1.0
    standardized = raw_scores / score_scale

    covariance_proxy = np.linalg.inv(system)
    graph_se = np.sqrt(np.maximum(np.diag(covariance_proxy), 0.0)) / score_scale
    information = np.diag(laplacian)
    return PairwiseRankingResult(
        scores={
            model_id: float(standardized[index[model_id]]) for model_id in model_ids
        },
        prior_scores={model_id: float(prior[index[model_id]]) for model_id in model_ids},
        edges=edges,
        graph_information={
            model_id: float(information[index[model_id]]) for model_id in model_ids
        },
        graph_standard_error={
            model_id: float(graph_se[index[model_id]]) for model_id in model_ids
        },
        score_scale=score_scale,
    )
