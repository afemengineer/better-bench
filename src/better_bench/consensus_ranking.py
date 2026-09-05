from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix


@dataclass(frozen=True)
class ConsensusRankingConfig:
    """Kemeny-style ranking over incomplete benchmark comparison panels.

    ``margin_temperature_points`` controls how quickly benchmark score margins saturate.
    A 30-point win should count as stronger evidence than a 1-point win, but should not
    be allowed to outweigh dozens of independent smaller wins merely because the raw
    metric has a large numerical spread.

    ``minimum_shared_families`` controls which direct model-pair preferences enter the
    consensus objective. Missing comparisons contribute no evidence in either direction.
    """

    margin_temperature_points: float = 5.0
    minimum_shared_families: int = 1
    information_scale_points: float = 10.0
    time_limit_seconds: float = 30.0


@dataclass(frozen=True)
class PairPreference:
    left_model: str
    right_model: str
    net_preference: float
    total_information: float
    shared_families: int
    shared_benchmarks: int


@dataclass(frozen=True)
class ConsensusRankingResult:
    ranking: list[str]
    rank: dict[str, int]
    scores: dict[str, float]
    preferences: list[PairPreference]
    weighted_agreement: float
    weighted_total: float
    solver_status: int
    solver_message: str


def _aggregate_model_benchmark_rows(
    rows: Iterable[Any],
) -> dict[tuple[str, str], tuple[float, float, str]]:
    grouped: dict[tuple[str, str], list[Any]] = defaultdict(list)
    for row in rows:
        grouped[(row.model_id, row.benchmark_id)].append(row)

    cells: dict[tuple[str, str], tuple[float, float, str]] = {}
    for key, values in grouped.items():
        weights = np.asarray([float(row.weight) for row in values], dtype=float)
        scores = np.asarray([float(row.score_points) for row in values], dtype=float)
        total = float(weights.sum())
        if total <= 1e-12:
            continue
        families = {str(row.family_id) for row in values}
        if len(families) != 1:
            raise ValueError(f"Benchmark {key[1]} maps to multiple families")
        cells[key] = (
            float(np.average(scores, weights=weights)),
            total,
            next(iter(families)),
        )
    return cells


def build_pair_preferences(
    rows: Iterable[Any],
    model_ids: list[str],
    state: Any,
    *,
    score_unit_points: float,
    config: ConsensusRankingConfig,
) -> list[PairPreference]:
    """Aggregate benchmark-local partial rankings into model-pair preferences.

    Benchmark difficulty cancels because only within-benchmark score differences enter.
    The comparison information uses the exact profiled weighted-least-squares factor
    ``w_i*w_k/W_j`` and the fixed estimator's learned general-factor discrimination.
    Family-adjusted benchmark weights are already embedded in the prepared row weights.
    """

    if config.margin_temperature_points <= 0:
        raise ValueError("margin_temperature_points must be positive")
    if config.minimum_shared_families < 1:
        raise ValueError("minimum_shared_families must be at least 1")
    if config.information_scale_points <= 0:
        raise ValueError("information_scale_points must be positive")

    general_values = np.asarray(
        [float(state.general[model_id]) for model_id in model_ids], dtype=float
    )
    general_scale = float(general_values.std(ddof=0))
    if general_scale <= 1e-12:
        general_scale = 1.0

    calibration: dict[str, float] = {}
    for benchmark_id, loading in state.loading.items():
        loading_points_per_z = float(score_unit_points * loading * general_scale)
        if loading_points_per_z <= 1e-8:
            continue
        calibration[str(benchmark_id)] = (
            loading_points_per_z / config.information_scale_points
        ) ** 2

    cells = _aggregate_model_benchmark_rows(rows)
    benchmark_total_weight: dict[str, float] = defaultdict(float)
    benchmarks_by_model: dict[str, set[str]] = defaultdict(set)
    for (model_id, benchmark_id), (_, weight, _) in cells.items():
        if benchmark_id not in calibration:
            continue
        benchmark_total_weight[benchmark_id] += weight
        benchmarks_by_model[model_id].add(benchmark_id)

    preferences: list[PairPreference] = []
    for left_index, left_model in enumerate(model_ids):
        for right_model in model_ids[left_index + 1 :]:
            shared = benchmarks_by_model[left_model] & benchmarks_by_model[right_model]
            by_family: dict[str, list[tuple[float, float]]] = defaultdict(list)
            for benchmark_id in shared:
                left_score, left_weight, family_id = cells[(left_model, benchmark_id)]
                right_score, right_weight, right_family_id = cells[
                    (right_model, benchmark_id)
                ]
                if family_id != right_family_id:
                    raise ValueError(
                        f"Benchmark {benchmark_id} has inconsistent family metadata"
                    )
                total_weight = benchmark_total_weight[benchmark_id]
                if total_weight <= 1e-12:
                    continue
                information = (
                    left_weight
                    * right_weight
                    / total_weight
                    * calibration[benchmark_id]
                )
                if information <= 1e-12:
                    continue
                margin = left_score - right_score
                bounded_margin = float(
                    np.tanh(margin / config.margin_temperature_points)
                )
                by_family[family_id].append((bounded_margin, information))

            if len(by_family) < config.minimum_shared_families:
                continue

            family_votes: list[float] = []
            family_information: list[float] = []
            benchmark_count = 0
            for values in by_family.values():
                benchmark_count += len(values)
                infos = np.asarray([value[1] for value in values], dtype=float)
                votes = np.asarray([value[0] for value in values], dtype=float)
                total_info = float(infos.sum())
                if total_info <= 1e-12:
                    continue
                family_votes.append(float(np.average(votes, weights=infos)))
                family_information.append(total_info)

            if len(family_votes) < config.minimum_shared_families:
                continue
            infos = np.asarray(family_information, dtype=float)
            votes = np.asarray(family_votes, dtype=float)
            total_info = float(infos.sum())
            if total_info <= 1e-12:
                continue
            preferences.append(
                PairPreference(
                    left_model=left_model,
                    right_model=right_model,
                    net_preference=float(np.sum(infos * votes)),
                    total_information=total_info,
                    shared_families=len(family_votes),
                    shared_benchmarks=benchmark_count,
                )
            )
    return preferences


def fit_consensus_ranking(
    rows: Iterable[Any],
    model_ids: list[str],
    state: Any,
    *,
    score_unit_points: float,
    config: ConsensusRankingConfig | None = None,
) -> ConsensusRankingResult:
    """Solve the weighted Kemeny consensus ranking exactly as a binary MILP.

    One binary variable is used per unordered model pair. Triangle constraints forbid
    cyclic triples, yielding a global total order that minimizes weighted disagreement
    with the incomplete benchmark-derived pair preferences.
    """

    config = config or ConsensusRankingConfig()
    if not model_ids:
        raise ValueError("model_ids cannot be empty")

    preferences = build_pair_preferences(
        rows,
        model_ids,
        state,
        score_unit_points=score_unit_points,
        config=config,
    )
    index = {model_id: idx for idx, model_id in enumerate(model_ids)}
    pair_to_var: dict[tuple[int, int], int] = {}
    pairs: list[tuple[int, int]] = []
    for left in range(len(model_ids)):
        for right in range(left + 1, len(model_ids)):
            pair_to_var[(left, right)] = len(pairs)
            pairs.append((left, right))

    objective = np.zeros(len(pairs), dtype=float)
    preference_by_pair: dict[tuple[int, int], PairPreference] = {}
    for preference in preferences:
        left = index[preference.left_model]
        right = index[preference.right_model]
        if left > right:
            left, right = right, left
            net = -preference.net_preference
        else:
            net = preference.net_preference
        var = pair_to_var[(left, right)]
        objective[var] = -net
        preference_by_pair[(left, right)] = preference

    constraint_rows: list[int] = []
    constraint_cols: list[int] = []
    constraint_data: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    row_number = 0
    n = len(model_ids)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                ij = pair_to_var[(i, j)]
                ik = pair_to_var[(i, k)]
                jk = pair_to_var[(j, k)]
                for column, value in ((ij, 1.0), (jk, 1.0), (ik, -1.0)):
                    constraint_rows.append(row_number)
                    constraint_cols.append(column)
                    constraint_data.append(value)
                lower.append(-np.inf)
                upper.append(1.0)
                row_number += 1

                for column, value in ((ij, -1.0), (jk, -1.0), (ik, 1.0)):
                    constraint_rows.append(row_number)
                    constraint_cols.append(column)
                    constraint_data.append(value)
                lower.append(-np.inf)
                upper.append(0.0)
                row_number += 1

    matrix = coo_matrix(
        (constraint_data, (constraint_rows, constraint_cols)),
        shape=(row_number, len(pairs)),
    ).tocsr()
    constraints = LinearConstraint(
        matrix,
        np.asarray(lower, dtype=float),
        np.asarray(upper, dtype=float),
    )
    result = milp(
        c=objective,
        integrality=np.ones(len(pairs), dtype=int),
        bounds=Bounds(np.zeros(len(pairs)), np.ones(len(pairs))),
        constraints=constraints,
        options={"time_limit": config.time_limit_seconds},
    )
    if result.x is None:
        raise RuntimeError(f"Consensus ranking solver failed: {result.message}")

    wins = np.zeros(n, dtype=int)
    for (left, right), variable in pair_to_var.items():
        if result.x[variable] >= 0.5:
            wins[left] += 1
        else:
            wins[right] += 1
    ranking_indices = sorted(range(n), key=lambda idx: (-wins[idx], model_ids[idx]))
    ranking = [model_ids[idx] for idx in ranking_indices]
    ranks = {model_id: rank for rank, model_id in enumerate(ranking, start=1)}

    weighted_agreement = 0.0
    weighted_total = 0.0
    for preference in preferences:
        magnitude = abs(preference.net_preference)
        if magnitude <= 1e-12:
            continue
        weighted_total += magnitude
        predicted_left = ranks[preference.left_model] < ranks[preference.right_model]
        observed_left = preference.net_preference > 0
        if predicted_left == observed_left:
            weighted_agreement += magnitude

    # Monotonic score used only as a convenient numeric ordering surface. The ordinal
    # ranking is the statistical object produced by Kemeny aggregation.
    if n == 1:
        scores = {ranking[0]: 0.0}
    else:
        midpoint = (n + 1) / 2.0
        denominator = max((n - 1) / 2.0, 1.0)
        scores = {
            model_id: float((midpoint - ranks[model_id]) / denominator)
            for model_id in model_ids
        }

    return ConsensusRankingResult(
        ranking=ranking,
        rank=ranks,
        scores=scores,
        preferences=preferences,
        weighted_agreement=weighted_agreement,
        weighted_total=weighted_total,
        solver_status=int(result.status),
        solver_message=str(result.message),
    )
