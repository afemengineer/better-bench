# ruff: noqa: I001
from __future__ import annotations

from collections import defaultdict
from datetime import date

import numpy as np

from better_bench.estimator import (
    EstimatorConfig,
    _balanced_folds,
    _fit_state,
    _general_scale,
    _prepare,
)
from better_bench.io import load_adoption, load_benchmarks, load_models, load_observations


AS_OF = date(2026, 9, 4)
FOLDS = 5
EDGE_MODES = ("empirical_std", "calibrated_loading")
MIN_SHARED_FAMILIES = (1, 2, 3, 4)
RIDGES = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
MARGINS = (0.0, 1.0, 3.0, 5.0, 10.0)


def _standardized_prior(state, model_ids):
    values = np.asarray([state.general[model_id] for model_id in model_ids], dtype=float)
    mean = float(values.mean())
    std = float(values.std(ddof=0))
    if std <= 1e-12:
        return np.zeros_like(values)
    return (values - mean) / std


def _benchmark_std(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row.benchmark_id].append(row)
    stats = {}
    for benchmark_id, group in grouped.items():
        values = np.asarray([row.score_points for row in group], dtype=float)
        weights = np.asarray([row.weight for row in group], dtype=float)
        if len(values) < 2 or float(weights.sum()) <= 1e-12:
            continue
        mean = float(np.average(values, weights=weights))
        variance = float(np.average(np.square(values - mean), weights=weights))
        std = float(np.sqrt(max(variance, 0.0)))
        if std > 1e-8:
            stats[benchmark_id] = std
    return stats


def _edge_calibration(rows, model_ids, state, config, mode):
    if mode == "empirical_std":
        stds = _benchmark_std(rows)
        return {
            benchmark_id: (std, 1.0)
            for benchmark_id, std in stds.items()
        }
    if mode != "calibrated_loading":
        raise ValueError(f"Unknown edge mode: {mode}")

    _, general_scale = _general_scale(state, model_ids)
    result = {}
    for benchmark_id, loading in state.loading.items():
        loading_points_per_z = (
            config.score_unit_points * loading * general_scale
        )
        if loading_points_per_z <= 1e-8:
            continue
        # Pairwise difference / loading is an implied difference in standardized g.
        # Fisher-style information scales with loading squared; normalize to a 10-point
        # loading so the graph ridge grid remains numerically interpretable.
        information_multiplier = (loading_points_per_z / 10.0) ** 2
        result[benchmark_id] = (loading_points_per_z, information_multiplier)
    return result


def _pairwise_edges(
    rows,
    model_ids,
    state,
    config,
    mode,
    minimum_shared_families,
):
    calibration = _edge_calibration(rows, model_ids, state, config, mode)
    by_model_benchmark = {
        (row.model_id, row.benchmark_id): row
        for row in rows
        if row.benchmark_id in calibration
    }
    benchmarks_by_model = defaultdict(set)
    for model_id, benchmark_id in by_model_benchmark:
        benchmarks_by_model[model_id].add(benchmark_id)

    edges = []
    for left_index, left_model in enumerate(model_ids):
        for right_model in model_ids[left_index + 1 :]:
            shared = benchmarks_by_model[left_model] & benchmarks_by_model[right_model]
            family_values = defaultdict(list)
            for benchmark_id in shared:
                left_row = by_model_benchmark[(left_model, benchmark_id)]
                right_row = by_model_benchmark[(right_model, benchmark_id)]
                scale, information_multiplier = calibration[benchmark_id]
                delta_z = (left_row.score_points - right_row.score_points) / scale
                base_weight = float(np.sqrt(left_row.weight * right_row.weight))
                pair_information = base_weight * information_multiplier
                family_values[left_row.family_id].append((delta_z, pair_information))
            if len(family_values) < minimum_shared_families:
                continue

            family_deltas = []
            family_weights = []
            for values in family_values.values():
                weights = np.asarray([item[1] for item in values], dtype=float)
                deltas = np.asarray([item[0] for item in values], dtype=float)
                total_weight = float(weights.sum())
                if total_weight <= 1e-12:
                    continue
                family_deltas.append(float(np.average(deltas, weights=weights)))
                # Multiple protocols from one family add sublinear information.
                family_weights.append(float(np.sqrt(total_weight)))
            if len(family_deltas) < minimum_shared_families:
                continue

            weights = np.asarray(family_weights, dtype=float)
            deltas = np.asarray(family_deltas, dtype=float)
            delta = float(np.average(deltas, weights=weights))
            edge_weight = float(weights.sum())
            edges.append((left_model, right_model, delta, edge_weight, len(family_deltas)))
    return edges


def _solve_graph(
    rows,
    model_ids,
    fixed_prior,
    state,
    config,
    mode,
    minimum_shared_families,
    ridge,
):
    index = {model_id: idx for idx, model_id in enumerate(model_ids)}
    edges = _pairwise_edges(
        rows,
        model_ids,
        state,
        config,
        mode,
        minimum_shared_families,
    )
    size = len(model_ids)
    laplacian = np.zeros((size, size), dtype=float)
    rhs = np.zeros(size, dtype=float)
    for left_model, right_model, delta, weight, _ in edges:
        left = index[left_model]
        right = index[right_model]
        laplacian[left, left] += weight
        laplacian[right, right] += weight
        laplacian[left, right] -= weight
        laplacian[right, left] -= weight
        rhs[left] += weight * delta
        rhs[right] -= weight * delta

    system = laplacian + ridge * np.eye(size)
    rhs = rhs + ridge * fixed_prior
    system += 1e-8 * np.eye(size)
    scores = np.linalg.solve(system, rhs)
    scores -= float(scores.mean())
    std = float(scores.std(ddof=0))
    if std > 1e-12:
        scores /= std
    return {model_id: float(scores[index[model_id]]) for model_id in model_ids}, edges


def _pairwise_counts(validation_rows, scores, margins=MARGINS):
    grouped = defaultdict(list)
    for row in validation_rows:
        grouped[row.benchmark_id].append(row)
    result = {}
    for margin in margins:
        correct = total = 0
        weighted_correct = weighted_total = 0.0
        for group in grouped.values():
            for left_index, left in enumerate(group):
                for right in group[left_index + 1 :]:
                    observed_delta = left.score_points - right.score_points
                    if abs(observed_delta) <= max(margin, 1e-9):
                        continue
                    predicted_delta = scores[left.model_id] - scores[right.model_id]
                    pair_weight = min(left.weight, right.weight)
                    total += 1
                    weighted_total += pair_weight
                    if observed_delta * predicted_delta > 0:
                        correct += 1
                        weighted_correct += pair_weight
        result[margin] = (correct, total, weighted_correct, weighted_total)
    return result


def _merge_counts(accumulators, fold_result):
    for margin, values in fold_result.items():
        for index, value in enumerate(values):
            accumulators[margin][index] += value


def _finalize(accumulators):
    result = {}
    for margin, (correct, total, weighted_correct, weighted_total) in accumulators.items():
        result[margin] = (
            correct / max(total, 1),
            weighted_correct / max(weighted_total, 1e-12),
            int(total),
        )
    return result


models = load_models("data/current")
benchmarks = load_benchmarks("data/current")
observations = load_observations("data/current")
adoption = load_adoption("data/current")
config = EstimatorConfig(jackknife_uncertainty=False)
rows, model_ids, benchmark_ids, _, _, _ = _prepare(
    models,
    benchmarks,
    observations,
    adoption,
    config,
    AS_OF,
)
assignment = _balanced_folds(rows, FOLDS)

results = {}
for mode in EDGE_MODES:
    for minimum_shared in MIN_SHARED_FAMILIES:
        for ridge in RIDGES:
            accumulators = {margin: [0, 0, 0.0, 0.0] for margin in MARGINS}
            for fold in range(FOLDS):
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
                fixed_state, _, _ = _fit_state(
                    training,
                    model_ids,
                    benchmark_ids,
                    config,
                )
                prior = _standardized_prior(fixed_state, model_ids)
                graph_scores, _ = _solve_graph(
                    training,
                    model_ids,
                    prior,
                    fixed_state,
                    config,
                    mode,
                    minimum_shared,
                    ridge,
                )
                _merge_counts(
                    accumulators,
                    _pairwise_counts(validation, graph_scores),
                )
            results[(mode, minimum_shared, ridge)] = _finalize(accumulators)

print(
    "mode\tmin_shared_families\tridge\tpair_acc\tweighted_pair_acc\t"
    "pair_acc_gt5\tweighted_pair_acc_gt5\tpairs"
)
for mode in EDGE_MODES:
    for minimum_shared in MIN_SHARED_FAMILIES:
        for ridge in RIDGES:
            row = results[(mode, minimum_shared, ridge)]
            acc0, weighted0, pairs0 = row[0.0]
            acc5, weighted5, _ = row[5.0]
            print(
                f"{mode}\t{minimum_shared}\t{ridge:.2f}\t{100*acc0:.2f}%\t"
                f"{100*weighted0:.2f}%\t{100*acc5:.2f}%\t"
                f"{100*weighted5:.2f}%\t{pairs0}"
            )


def _objective(item):
    result = item[1]
    return float(np.mean([result[margin][0] for margin in MARGINS]))


best_key, best_result = max(results.items(), key=_objective)
print(
    f"best_exploratory=mode_{best_key[0]}_min_shared{best_key[1]}_ridge{best_key[2]:.2f} "
    f"mean_margin_accuracy={100*_objective((best_key, best_result)):.2f}%"
)

full_state, _, _ = _fit_state(rows, model_ids, benchmark_ids, config)
full_prior = _standardized_prior(full_state, model_ids)
full_scores, full_edges = _solve_graph(
    rows,
    model_ids,
    full_prior,
    full_state,
    config,
    best_key[0],
    best_key[1],
    best_key[2],
)
ranking = sorted(full_scores.items(), key=lambda item: item[1], reverse=True)
print("rank\tmodel\tpairwise_graph_z\tfixed_prior_z")
prior_map = {model_id: float(full_prior[index]) for index, model_id in enumerate(model_ids)}
for rank, (model_id, score) in enumerate(ranking, start=1):
    print(f"{rank}\t{model_id}\t{score:+.4f}\t{prior_map[model_id]:+.4f}")

edge_lookup = {
    frozenset((left, right)): (left, right, delta, weight, families)
    for left, right, delta, weight, families in full_edges
}
key = frozenset(("glm-5.3", "glm-5.3-flash"))
if key in edge_lookup:
    left, _, delta, weight, families = edge_lookup[key]
    oriented = delta if left == "glm-5.3" else -delta
    print(
        "glm_pair_edge\t"
        f"delta_z={oriented:+.4f}\tweight={weight:.3f}\tshared_families={families}\t"
        f"graph_delta={full_scores['glm-5.3']-full_scores['glm-5.3-flash']:+.4f}"
    )

print(
    "note=exploratory hyperparameters use the same family-CV folds; require nested or "
    "stability validation before promoting a graph score to production BBI"
)
