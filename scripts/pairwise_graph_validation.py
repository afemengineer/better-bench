# ruff: noqa: I001
from __future__ import annotations

from collections import defaultdict
from datetime import date

import numpy as np
from scipy.stats import spearmanr

from better_bench.estimator import (
    EstimatorConfig,
    _balanced_folds,
    _fit_state,
    _prepare,
)
from better_bench.io import load_adoption, load_benchmarks, load_models, load_observations
from better_bench.pairwise_ranking import (
    PairwiseRankingConfig,
    build_pairwise_edges,
    fit_pairwise_ranking,
)


AS_OF = date(2026, 9, 4)
OUTER_FOLDS = 5
INNER_FOLDS = 4
FRONTIER_SIZE = 10
MARGINS = (0.0, 1.0, 3.0, 5.0, 10.0)
CANDIDATES = tuple(
    PairwiseRankingConfig(minimum_shared_families=minimum_shared, ridge=ridge)
    for minimum_shared in (1, 2, 3, 4)
    for ridge in (0.3, 1.0, 3.0, 10.0)
)


def _standardized_fixed_scores(state, model_ids):
    values = np.asarray([state.general[model_id] for model_id in model_ids], dtype=float)
    values -= float(values.mean())
    scale = float(values.std(ddof=0))
    if scale <= 1e-12:
        scale = 1.0
    values /= scale
    return {model_id: float(values[index]) for index, model_id in enumerate(model_ids)}


def _pair_counts(rows, scores, *, allowed_models=None):
    grouped = defaultdict(list)
    for row in rows:
        if allowed_models is not None and row.model_id not in allowed_models:
            continue
        grouped[row.benchmark_id].append(row)

    result = {margin: [0, 0, 0.0, 0.0] for margin in MARGINS}
    for margin in MARGINS:
        for group in grouped.values():
            for left_index, left in enumerate(group):
                for right in group[left_index + 1 :]:
                    observed_delta = left.score_points - right.score_points
                    if abs(observed_delta) <= max(margin, 1e-9):
                        continue
                    predicted_delta = scores[left.model_id] - scores[right.model_id]
                    pair_weight = min(left.weight, right.weight)
                    result[margin][1] += 1
                    result[margin][3] += pair_weight
                    if observed_delta * predicted_delta > 0:
                        result[margin][0] += 1
                        result[margin][2] += pair_weight
    return result


def _merge(target, source):
    for margin in MARGINS:
        for index, value in enumerate(source[margin]):
            target[margin][index] += value


def _rates(counts):
    result = {}
    for margin, (correct, total, weighted_correct, weighted_total) in counts.items():
        result[margin] = (
            correct / max(total, 1),
            weighted_correct / max(weighted_total, 1e-12),
            int(total),
        )
    return result


def _objective(counts):
    rates = _rates(counts)
    return float(np.mean([rates[margin][0] for margin in MARGINS]))


def _cross_validated_counts(rows, model_ids, benchmark_ids, estimator_config, candidate, folds):
    assignment = _balanced_folds(rows, folds)
    counts = {margin: [0, 0, 0.0, 0.0] for margin in MARGINS}
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
        state, _, _ = _fit_state(training, model_ids, benchmark_ids, estimator_config)
        graph = fit_pairwise_ranking(
            training,
            model_ids,
            state,
            score_unit_points=estimator_config.score_unit_points,
            config=candidate,
        )
        _merge(counts, _pair_counts(validation, graph.scores))
    return counts


def _select_candidate(rows, model_ids, benchmark_ids, estimator_config, folds):
    scored = []
    for candidate in CANDIDATES:
        counts = _cross_validated_counts(
            rows,
            model_ids,
            benchmark_ids,
            estimator_config,
            candidate,
            folds,
        )
        scored.append((_objective(counts), candidate, counts))
    scored.sort(
        key=lambda item: (
            item[0],
            -item[1].minimum_shared_families,
            -item[1].ridge,
        ),
        reverse=True,
    )
    return scored[0], scored


models = load_models("data/current")
benchmarks = load_benchmarks("data/current")
observations = load_observations("data/current")
adoption = load_adoption("data/current")
estimator_config = EstimatorConfig(jackknife_uncertainty=False)
rows, model_ids, benchmark_ids, _, _, _ = _prepare(
    models,
    benchmarks,
    observations,
    adoption,
    estimator_config,
    AS_OF,
)
outer_assignment = _balanced_folds(rows, OUTER_FOLDS)

fixed_total = {margin: [0, 0, 0.0, 0.0] for margin in MARGINS}
graph_total = {margin: [0, 0, 0.0, 0.0] for margin in MARGINS}
fixed_frontier_total = {margin: [0, 0, 0.0, 0.0] for margin in MARGINS}
graph_frontier_total = {margin: [0, 0, 0.0, 0.0] for margin in MARGINS}
selected_configs = []

print("nested_outer_fold\tselected_min_shared\tselected_ridge\tinner_objective\touter_pair_acc\touter_graph_acc\tfrontier_fixed_acc\tfrontier_graph_acc")
for outer_fold in range(OUTER_FOLDS):
    outer_training = [
        row
        for row in rows
        if outer_assignment[(row.model_id, row.family_id)] != outer_fold
    ]
    outer_validation = [
        row
        for row in rows
        if outer_assignment[(row.model_id, row.family_id)] == outer_fold
    ]
    best, _ = _select_candidate(
        outer_training,
        model_ids,
        benchmark_ids,
        estimator_config,
        INNER_FOLDS,
    )
    inner_objective, candidate, _ = best
    selected_configs.append(candidate)

    state, _, _ = _fit_state(
        outer_training,
        model_ids,
        benchmark_ids,
        estimator_config,
    )
    fixed_scores = _standardized_fixed_scores(state, model_ids)
    graph = fit_pairwise_ranking(
        outer_training,
        model_ids,
        state,
        score_unit_points=estimator_config.score_unit_points,
        config=candidate,
    )
    frontier = set(
        sorted(model_ids, key=lambda model_id: fixed_scores[model_id], reverse=True)[
            :FRONTIER_SIZE
        ]
    )

    fixed_counts = _pair_counts(outer_validation, fixed_scores)
    graph_counts = _pair_counts(outer_validation, graph.scores)
    fixed_frontier = _pair_counts(
        outer_validation,
        fixed_scores,
        allowed_models=frontier,
    )
    graph_frontier = _pair_counts(
        outer_validation,
        graph.scores,
        allowed_models=frontier,
    )
    _merge(fixed_total, fixed_counts)
    _merge(graph_total, graph_counts)
    _merge(fixed_frontier_total, fixed_frontier)
    _merge(graph_frontier_total, graph_frontier)

    fixed_rate = _rates(fixed_counts)[0.0][0]
    graph_rate = _rates(graph_counts)[0.0][0]
    frontier_fixed_rate = _rates(fixed_frontier)[0.0][0]
    frontier_graph_rate = _rates(graph_frontier)[0.0][0]
    print(
        f"{outer_fold}\t{candidate.minimum_shared_families}\t{candidate.ridge:.2f}\t"
        f"{100*inner_objective:.2f}%\t{100*fixed_rate:.2f}%\t{100*graph_rate:.2f}%\t"
        f"{100*frontier_fixed_rate:.2f}%\t{100*frontier_graph_rate:.2f}%"
    )

print("nested_summary\tmargin\tfixed_acc\tgraph_acc\tfixed_weighted\tgraph_weighted\tpairs")
fixed_rates = _rates(fixed_total)
graph_rates = _rates(graph_total)
for margin in MARGINS:
    fixed_acc, fixed_weighted, pairs = fixed_rates[margin]
    graph_acc, graph_weighted, _ = graph_rates[margin]
    print(
        f"nested_summary\t{margin:.1f}\t{100*fixed_acc:.2f}%\t{100*graph_acc:.2f}%\t"
        f"{100*fixed_weighted:.2f}%\t{100*graph_weighted:.2f}%\t{pairs}"
    )

print("frontier_summary\tmargin\tfixed_acc\tgraph_acc\tfixed_weighted\tgraph_weighted\tpairs")
fixed_frontier_rates = _rates(fixed_frontier_total)
graph_frontier_rates = _rates(graph_frontier_total)
for margin in MARGINS:
    fixed_acc, fixed_weighted, pairs = fixed_frontier_rates[margin]
    graph_acc, graph_weighted, _ = graph_frontier_rates[margin]
    print(
        f"frontier_summary\t{margin:.1f}\t{100*fixed_acc:.2f}%\t{100*graph_acc:.2f}%\t"
        f"{100*fixed_weighted:.2f}%\t{100*graph_weighted:.2f}%\t{pairs}"
    )

# Select the deployment hyperparameters on all available data. Nested CV above is the
# unbiased estimate of selection performance; this full-data selection is only the
# deployment choice.
full_best, full_grid = _select_candidate(
    rows,
    model_ids,
    benchmark_ids,
    estimator_config,
    OUTER_FOLDS,
)
full_objective, deployment_config, deployment_counts = full_best
print(
    "deployment_selection\t"
    f"min_shared={deployment_config.minimum_shared_families}\t"
    f"ridge={deployment_config.ridge:.2f}\t"
    f"mean_margin_accuracy={100*full_objective:.2f}%"
)

print("deployment_grid\tmin_shared\tridge\tmean_margin_acc\tall_acc\tgt5_acc")
for objective, candidate, counts in full_grid:
    rates = _rates(counts)
    print(
        f"deployment_grid\t{candidate.minimum_shared_families}\t{candidate.ridge:.2f}\t"
        f"{100*objective:.2f}%\t{100*rates[0.0][0]:.2f}%\t{100*rates[5.0][0]:.2f}%"
    )

full_state, _, _ = _fit_state(rows, model_ids, benchmark_ids, estimator_config)
fixed_scores = _standardized_fixed_scores(full_state, model_ids)
full_graph = fit_pairwise_ranking(
    rows,
    model_ids,
    full_state,
    score_unit_points=estimator_config.score_unit_points,
    config=deployment_config,
)
ranking = sorted(model_ids, key=lambda model_id: full_graph.scores[model_id], reverse=True)
print("deployment_ranking\trank\tmodel\tgraph_z\tfixed_z\tgraph_information\tgraph_se")
for rank, model_id in enumerate(ranking, start=1):
    print(
        f"deployment_ranking\t{rank}\t{model_id}\t{full_graph.scores[model_id]:+.4f}\t"
        f"{fixed_scores[model_id]:+.4f}\t{full_graph.graph_information[model_id]:.3f}\t"
        f"{full_graph.graph_standard_error[model_id]:.4f}"
    )
print(
    "glm_deployment_delta\t"
    f"{full_graph.scores['glm-5.3']-full_graph.scores['glm-5.3-flash']:+.4f}"
)

# Pairwise contradiction audit on a stricter four-family direct-evidence panel.
diagnostic_config = PairwiseRankingConfig(minimum_shared_families=4, ridge=1.0)
direct_edges = build_pairwise_edges(
    rows,
    model_ids,
    full_state,
    score_unit_points=estimator_config.score_unit_points,
    config=diagnostic_config,
)
fixed_inversions = 0
graph_inversions = 0
for edge in direct_edges:
    if abs(edge.delta_z) <= 1e-12:
        continue
    fixed_delta = fixed_scores[edge.left_model] - fixed_scores[edge.right_model]
    graph_delta = full_graph.scores[edge.left_model] - full_graph.scores[edge.right_model]
    fixed_inversions += int(fixed_delta * edge.delta_z < 0)
    graph_inversions += int(graph_delta * edge.delta_z < 0)
print(
    f"direct_evidence_inversions\tpairs={len(direct_edges)}\t"
    f"fixed={fixed_inversions}\tgraph={graph_inversions}"
)

# Family perturbation stability. Omit every independent family, refit calibration and
# graph, then measure whether the ranking and the GLM ordering are stable.
full_score_vector = np.asarray([full_graph.scores[model_id] for model_id in model_ids])
full_top10 = set(ranking[:10])
family_ids = sorted({row.family_id for row in rows})
stability_rho = []
stability_top10 = []
glm_main_wins = 0
for family_id in family_ids:
    reduced = [row for row in rows if row.family_id != family_id]
    reduced_state, _, _ = _fit_state(
        reduced,
        model_ids,
        benchmark_ids,
        estimator_config,
    )
    reduced_graph = fit_pairwise_ranking(
        reduced,
        model_ids,
        reduced_state,
        score_unit_points=estimator_config.score_unit_points,
        config=deployment_config,
    )
    vector = np.asarray([reduced_graph.scores[model_id] for model_id in model_ids])
    rho = spearmanr(full_score_vector, vector).statistic
    if np.isfinite(rho):
        stability_rho.append(float(rho))
    reduced_ranking = sorted(
        model_ids,
        key=lambda model_id: reduced_graph.scores[model_id],
        reverse=True,
    )
    stability_top10.append(len(full_top10 & set(reduced_ranking[:10])) / 10.0)
    if reduced_graph.scores["glm-5.3"] > reduced_graph.scores["glm-5.3-flash"]:
        glm_main_wins += 1
print(
    f"family_stability\tomissions={len(family_ids)}\t"
    f"median_spearman={np.median(stability_rho):.4f}\t"
    f"min_spearman={np.min(stability_rho):.4f}\t"
    f"median_top10_overlap={100*np.median(stability_top10):.1f}%\t"
    f"glm_main_over_flash={glm_main_wins}/{len(family_ids)}"
)

print(
    "note=nested outer CV estimates ranking performance after hyperparameter selection; "
    "frontier membership is determined from training-only fixed scores to avoid test leakage"
)
