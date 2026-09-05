# ruff: noqa: I001
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import date

import numpy as np

from better_bench.estimator import EstimatorConfig, _balanced_folds, _prepare
from better_bench.io import load_adoption, load_benchmarks, load_models, load_observations


AS_OF = date(2026, 9, 4)
FOLDS = 5
EPS = 0.5
MARGINS = (0.0, 1.0, 3.0, 5.0, 10.0)
_TIER_FACTOR = {
    "core": 1.00,
    "high_value_emerging": 0.90,
    "supporting": 0.55,
    "diagnostic_only": 0.20,
}


@dataclass(frozen=True)
class CompletionConfig:
    rank: int
    lam: float
    iterations: int = 50
    ensembles: int = 5


def _logit(score: np.ndarray) -> np.ndarray:
    p = np.clip(score, EPS, 100.0 - EPS) / 100.0
    return np.log(p / (1.0 - p))


def _inv_logit(value: np.ndarray) -> np.ndarray:
    return 100.0 / (1.0 + np.exp(-value))


def _fit_bias_als(
    matrix: np.ndarray,
    config: CompletionConfig,
    *,
    base_seed: int = 42,
) -> np.ndarray:
    """BenchPress-style logit + column-z-score + bias-decomposed rank-r ALS."""
    observed = np.isfinite(matrix)
    transformed = np.full_like(matrix, np.nan, dtype=float)
    transformed[observed] = _logit(matrix[observed])

    col_mean = np.nanmean(transformed, axis=0)
    col_std = np.nanstd(transformed, axis=0)
    col_std = np.where(np.isfinite(col_std) & (col_std > 1e-8), col_std, 1.0)
    z = (transformed - col_mean[None, :]) / col_std[None, :]

    n_models, n_bench = matrix.shape
    row_obs = [np.where(observed[i])[0] for i in range(n_models)]
    col_obs = [np.where(observed[:, j])[0] for j in range(n_bench)]
    obs_ij = np.argwhere(observed)
    obs_values = z[observed]
    global_mean = float(np.mean(obs_values)) if obs_values.size else 0.0
    eye = np.eye(config.rank + 1) * config.lam

    def run_one(seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        mu = global_mean
        model_bias = np.zeros(n_models)
        bench_bias = np.zeros(n_bench)
        u = rng.normal(0.0, 0.01, size=(n_models, config.rank))
        v = rng.normal(0.0, 0.01, size=(n_bench, config.rank))

        for _ in range(config.iterations):
            for i in range(n_models):
                js = row_obs[i]
                if js.size == 0:
                    continue
                residual = z[i, js] - mu - bench_bias[js]
                design = np.column_stack((np.ones(js.size), v[js]))
                solution = np.linalg.solve(design.T @ design + eye, design.T @ residual)
                model_bias[i] = solution[0]
                u[i] = solution[1:]

            for j in range(n_bench):
                is_ = col_obs[j]
                if is_.size == 0:
                    continue
                residual = z[is_, j] - mu - model_bias[is_]
                design = np.column_stack((np.ones(is_.size), u[is_]))
                solution = np.linalg.solve(design.T @ design + eye, design.T @ residual)
                bench_bias[j] = solution[0]
                v[j] = solution[1:]

            interaction = np.einsum(
                "ij,ij->i",
                u[obs_ij[:, 0]],
                v[obs_ij[:, 1]],
            )
            mu = float(
                np.mean(
                    obs_values
                    - model_bias[obs_ij[:, 0]]
                    - bench_bias[obs_ij[:, 1]]
                    - interaction
                )
            )

        return mu + model_bias[:, None] + bench_bias[None, :] + u @ v.T

    prediction_z = np.zeros_like(matrix, dtype=float)
    for ensemble in range(config.ensembles):
        prediction_z += run_one(base_seed + ensemble)
    prediction_z /= config.ensembles
    prediction_logit = prediction_z * col_std[None, :] + col_mean[None, :]
    return np.clip(_inv_logit(prediction_logit), 0.0, 100.0)


def _matrix_from_rows(rows, model_ids, benchmark_ids, *, hide_fold=None, assignment=None):
    model_index = {model_id: i for i, model_id in enumerate(model_ids)}
    bench_index = {benchmark_id: j for j, benchmark_id in enumerate(benchmark_ids)}
    sums = np.zeros((len(model_ids), len(benchmark_ids)), dtype=float)
    weights = np.zeros_like(sums)
    for row in rows:
        if hide_fold is not None and assignment[(row.model_id, row.family_id)] == hide_fold:
            continue
        i = model_index[row.model_id]
        j = bench_index[row.benchmark_id]
        sums[i, j] += row.weight * row.score_points
        weights[i, j] += row.weight
    matrix = np.full_like(sums, np.nan)
    mask = weights > 0
    matrix[mask] = sums[mask] / weights[mask]
    return matrix


def _common_panel_scores(training, prediction, panel_weights):
    """Score every model on the same benchmark panel.

    Observed cells are retained; only missing cells are imputed. Each benchmark is
    standardized in training-only logit space before aggregation so benchmark difficulty
    and raw score scale cannot reward a model merely because it was tested on an easier
    portfolio.
    """
    observed = np.isfinite(training)
    completed = np.where(observed, training, prediction)
    transformed_training = np.full_like(training, np.nan, dtype=float)
    transformed_training[observed] = _logit(training[observed])
    col_mean = np.nanmean(transformed_training, axis=0)
    col_std = np.nanstd(transformed_training, axis=0)
    valid_columns = np.isfinite(col_mean) & np.isfinite(col_std) & (col_std > 1e-8)

    completed_logit = _logit(completed)
    z = np.zeros_like(completed_logit)
    z[:, valid_columns] = (
        completed_logit[:, valid_columns] - col_mean[None, valid_columns]
    ) / col_std[None, valid_columns]
    weights = np.asarray(panel_weights, dtype=float) * valid_columns.astype(float)
    if float(weights.sum()) <= 1e-12:
        raise ValueError("No valid benchmark weights for common-panel scoring")
    return (z * weights[None, :]).sum(axis=1) / weights.sum()


def _pair_counts(heldout_by_benchmark, ranking_score, margins=MARGINS):
    result = {}
    for margin in margins:
        correct = total = 0
        weighted_correct = weighted_total = 0.0
        for values in heldout_by_benchmark.values():
            for left in range(len(values)):
                for right in range(left + 1, len(values)):
                    observed_delta = values[left][1] - values[right][1]
                    if abs(observed_delta) <= max(margin, 1e-9):
                        continue
                    predicted_delta = (
                        ranking_score[values[left][0]] - ranking_score[values[right][0]]
                    )
                    pair_weight = min(values[left][3], values[right][3])
                    total += 1
                    weighted_total += pair_weight
                    if observed_delta * predicted_delta > 0:
                        correct += 1
                        weighted_correct += pair_weight
        result[margin] = (correct, total, weighted_correct, weighted_total)
    return result


def _evaluate(
    config: CompletionConfig,
    rows,
    model_ids,
    benchmark_ids,
    assignment,
    panel_weights,
):
    model_index = {model_id: i for i, model_id in enumerate(model_ids)}
    bench_index = {benchmark_id: j for j, benchmark_id in enumerate(benchmark_ids)}
    squared = absolute = total_weight = 0.0
    baseline_squared = 0.0
    observed_values = []
    predicted_values = []
    cell_pair_correct = cell_pair_total = 0
    panel_counts = {margin: [0, 0, 0.0, 0.0] for margin in MARGINS}

    for fold in range(FOLDS):
        training = _matrix_from_rows(
            rows,
            model_ids,
            benchmark_ids,
            hide_fold=fold,
            assignment=assignment,
        )
        prediction = _fit_bias_als(training, config, base_seed=42 + fold * 100)
        panel_score_array = _common_panel_scores(training, prediction, panel_weights)
        panel_score = {
            model_id: float(panel_score_array[index])
            for index, model_id in enumerate(model_ids)
        }
        col_mean = np.nanmean(training, axis=0)

        heldout_by_benchmark = defaultdict(list)
        for row in rows:
            if assignment[(row.model_id, row.family_id)] != fold:
                continue
            i = model_index[row.model_id]
            j = bench_index[row.benchmark_id]
            if not np.isfinite(col_mean[j]) or not np.isfinite(prediction[i, j]):
                continue
            error = row.score_points - prediction[i, j]
            baseline_error = row.score_points - col_mean[j]
            squared += row.weight * error * error
            baseline_squared += row.weight * baseline_error * baseline_error
            absolute += row.weight * abs(error)
            total_weight += row.weight
            observed_values.append(row.score_points)
            predicted_values.append(prediction[i, j])
            heldout_by_benchmark[row.benchmark_id].append(
                (row.model_id, row.score_points, prediction[i, j], row.weight)
            )

        for values in heldout_by_benchmark.values():
            for left in range(len(values)):
                for right in range(left + 1, len(values)):
                    observed_delta = values[left][1] - values[right][1]
                    predicted_delta = values[left][2] - values[right][2]
                    if abs(observed_delta) <= 1e-9:
                        continue
                    cell_pair_total += 1
                    if observed_delta * predicted_delta > 0:
                        cell_pair_correct += 1

        fold_counts = _pair_counts(heldout_by_benchmark, panel_score)
        for margin, values in fold_counts.items():
            for index, value in enumerate(values):
                panel_counts[margin][index] += value

    rmse = math.sqrt(squared / max(total_weight, 1e-12))
    baseline_rmse = math.sqrt(baseline_squared / max(total_weight, 1e-12))
    mae = absolute / max(total_weight, 1e-12)
    corr = float(np.corrcoef(observed_values, predicted_values)[0, 1])
    panel_accuracy = {}
    for margin, (correct, total, weighted_correct, weighted_total) in panel_counts.items():
        panel_accuracy[margin] = (
            correct / max(total, 1),
            weighted_correct / max(weighted_total, 1e-12),
            int(total),
        )
    return (
        rmse,
        baseline_rmse,
        mae,
        corr,
        cell_pair_correct / max(cell_pair_total, 1),
        cell_pair_total,
        panel_accuracy,
    )


models = load_models("data/current")
benchmarks = load_benchmarks("data/current")
observations = load_observations("data/current")
adoption = load_adoption("data/current")
estimator_config = EstimatorConfig(jackknife_uncertainty=False)
(
    rows,
    model_ids,
    benchmark_ids,
    benchmark_by_id,
    evidence_weight,
    tiers,
) = _prepare(
    models,
    benchmarks,
    observations,
    adoption,
    estimator_config,
    AS_OF,
)
assignment = _balanced_folds(rows, FOLDS)
panel_weights = np.asarray(
    [
        evidence_weight[benchmark_id] * _TIER_FACTOR[tiers[benchmark_id]]
        for benchmark_id in benchmark_ids
    ],
    dtype=float,
)

candidates = [
    CompletionConfig(rank=rank, lam=lam)
    for rank in (1, 2, 3, 4)
    for lam in (0.03, 0.10, 0.30, 1.00)
]
print(
    "exploratory=true note=hyperparameter grid is evaluated on the same family folds; "
    "use nested CV before promoting a completion model into production scoring"
)
print(
    "rank\tlambda\tweighted_RMSE\tbenchmark_mean_RMSE\tweighted_MAE\tcorrelation\t"
    "cell_pair_acc\tpanel_pair_acc\tpanel_weighted_acc\tpanel_pair_acc_gt5\tpairs"
)
results = []
for candidate in candidates:
    metrics = _evaluate(
        candidate,
        rows,
        model_ids,
        benchmark_ids,
        assignment,
        panel_weights,
    )
    panel0 = metrics[6][0.0]
    panel5 = metrics[6][5.0]
    results.append((candidate, metrics))
    print(
        f"{candidate.rank}\t{candidate.lam:.2f}\t{metrics[0]:.3f}\t{metrics[1]:.3f}\t"
        f"{metrics[2]:.3f}\t{metrics[3]:+.3f}\t{100 * metrics[4]:.1f}%\t"
        f"{100 * panel0[0]:.2f}%\t{100 * panel0[1]:.2f}%\t"
        f"{100 * panel5[0]:.2f}%\t{panel0[2]}"
    )

best_rmse = min(results, key=lambda item: item[1][0])
best_rank = max(
    results,
    key=lambda item: float(
        np.mean([item[1][6][margin][0] for margin in MARGINS])
    ),
)
print(
    f"best_rmse_exploratory=rank{best_rmse[0].rank}_lambda{best_rmse[0].lam:.2f}"
)
print(
    f"best_ranking_exploratory=rank{best_rank[0].rank}_lambda{best_rank[0].lam:.2f} "
    f"mean_margin_accuracy={100*np.mean([best_rank[1][6][margin][0] for margin in MARGINS]):.2f}%"
)

best = best_rank[0]
full_matrix = _matrix_from_rows(rows, model_ids, benchmark_ids)
full_prediction = _fit_bias_als(full_matrix, best)
full_panel = _common_panel_scores(full_matrix, full_prediction, panel_weights)
full_panel = (full_panel - full_panel.mean()) / max(float(full_panel.std(ddof=0)), 1e-12)
model_index = {model_id: i for i, model_id in enumerate(model_ids)}
bench_index = {benchmark_id: j for j, benchmark_id in enumerate(benchmark_ids)}

print("common_panel_ranking\trank\tmodel\tpanel_z")
for rank, index in enumerate(np.argsort(-full_panel), start=1):
    print(f"common_panel_ranking\t{rank}\t{model_ids[index]}\t{full_panel[index]:+.4f}")
print(
    "glm_common_panel_delta\t"
    f"{full_panel[model_index['glm-5.3']]-full_panel[model_index['glm-5.3-flash']]:+.4f}"
)

print("glm_flash_missing_sensitivity\tbenchmark\tfamily\tpredicted_normalized_score\tmain_observed")
for benchmark_id in (
    "frontierswe-v2-2026-09",
    "terminal-bench-4.0",
    "terminal-bench-science-0.1",
):
    if benchmark_id not in bench_index:
        continue
    flash_i = model_index.get("glm-5.3-flash")
    main_i = model_index.get("glm-5.3")
    j = bench_index[benchmark_id]
    if flash_i is None or main_i is None:
        continue
    main_score = full_matrix[main_i, j]
    family = benchmark_by_id[benchmark_id].family_id or benchmark_id
    print(
        f"glm-5.3-flash\t{benchmark_id}\t{family}\t{full_prediction[flash_i, j]:.2f}\t"
        f"{main_score:.2f}"
    )
