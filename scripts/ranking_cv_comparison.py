# ruff: noqa: I001
from __future__ import annotations

from collections import defaultdict
from datetime import date

import numpy as np

from better_bench.estimator import (
    EstimatorConfig,
    _balanced_folds,
    _fit_state as _fit_fixed_state,
    _prepare as _prepare_fixed,
)
from better_bench.hierarchical import (
    HierarchicalConfig,
    _centered_capability_loadings,
    _prediction as _hierarchical_prediction,
    _prepare as _prepare_hierarchical,
)
from better_bench.hierarchical_validation import (
    _fit_state as _fit_hierarchical_state,
    _raw_normalized_scores,
    _standardize_rows,
    _training_standardization,
)
from better_bench.io import load_adoption, load_benchmarks, load_models, load_observations


AS_OF = date(2026, 9, 4)
FOLDS = 5
MARGINS = (0.0, 1.0, 3.0, 5.0, 10.0)


def _pairwise_accuracy(records, margins=MARGINS):
    # records: (fold, benchmark_id, model_id, observed_raw_points, predicted_order_score, weight)
    grouped = defaultdict(list)
    for record in records:
        grouped[(record[0], record[1])].append(record)

    result = {}
    for margin in margins:
        correct = total = 0
        weighted_correct = weighted_total = 0.0
        for values in grouped.values():
            for left in range(len(values)):
                for right in range(left + 1, len(values)):
                    obs_delta = values[left][3] - values[right][3]
                    if abs(obs_delta) <= max(margin, 1e-9):
                        continue
                    pred_delta = values[left][4] - values[right][4]
                    pair_weight = min(values[left][5], values[right][5])
                    total += 1
                    weighted_total += pair_weight
                    if obs_delta * pred_delta > 0:
                        correct += 1
                        weighted_correct += pair_weight
        result[margin] = (
            correct / max(total, 1),
            weighted_correct / max(weighted_total, 1e-12),
            total,
        )
    return result


models = load_models("data/current")
benchmarks = load_benchmarks("data/current")
observations = load_observations("data/current")
adoption = load_adoption("data/current")

fixed_config = EstimatorConfig(jackknife_uncertainty=False)
(
    fixed_rows,
    model_ids,
    benchmark_ids,
    benchmark_by_id,
    _,
    _,
) = _prepare_fixed(models, benchmarks, observations, adoption, fixed_config, AS_OF)
assignment = _balanced_folds(fixed_rows, FOLDS)

hier_config = HierarchicalConfig(
    minimum_models_per_benchmark=fixed_config.minimum_models_per_benchmark,
    minimum_benchmarks_per_model=fixed_config.minimum_benchmarks_per_model,
    minimum_families_per_model=fixed_config.minimum_families_per_model,
)
(
    hier_rows,
    hier_frame,
    hier_benchmark_by_id,
    _,
    hier_quality,
) = _prepare_hierarchical(models, benchmarks, observations, adoption, hier_config, AS_OF)
hier_model_ids = [str(item) for item in hier_frame.index]
hier_benchmark_ids = [str(item) for item in hier_frame.columns]
if hier_model_ids != model_ids or hier_benchmark_ids != benchmark_ids:
    raise ValueError("Fixed and hierarchical estimators retained different matrices")
_, centered_loadings = _centered_capability_loadings(
    benchmark_ids,
    hier_benchmark_by_id,
    hier_quality,
)
retained_pairs = {(row.model_id, row.benchmark_id) for row in hier_rows}
raw_scores = _raw_normalized_scores(observations, benchmark_by_id, retained_pairs)

fixed_records = []
hier_records = []
for fold in range(FOLDS):
    fixed_training = [
        row for row in fixed_rows if assignment[(row.model_id, row.family_id)] != fold
    ]
    fixed_validation = [
        row for row in fixed_rows if assignment[(row.model_id, row.family_id)] == fold
    ]
    fixed_state, _, _ = _fit_fixed_state(
        fixed_training,
        model_ids,
        benchmark_ids,
        fixed_config,
    )
    for row in fixed_validation:
        prediction = fixed_config.score_unit_points * (
            fixed_state.intercept[row.benchmark_id]
            + fixed_state.loading[row.benchmark_id] * fixed_state.general[row.model_id]
        )
        fixed_records.append(
            (
                fold,
                row.benchmark_id,
                row.model_id,
                row.score_points,
                prediction,
                row.weight,
            )
        )

    hier_training_base = [
        row for row in hier_rows if assignment[(row.model_id, row.family_id)] != fold
    ]
    hier_validation_base = [
        row for row in hier_rows if assignment[(row.model_id, row.family_id)] == fold
    ]
    stats = _training_standardization(hier_training_base, raw_scores)
    hier_training = _standardize_rows(hier_training_base, raw_scores, stats)
    hier_validation = _standardize_rows(hier_validation_base, raw_scores, stats)
    hier_state = _fit_hierarchical_state(
        hier_training,
        model_ids,
        benchmark_ids,
        centered_loadings,
        hier_config,
    )
    for row in hier_validation:
        raw = raw_scores[(row.model_id, row.benchmark_id)]
        prediction = _hierarchical_prediction(row, hier_state, centered_loadings)
        hier_records.append(
            (
                fold,
                row.benchmark_id,
                row.model_id,
                raw,
                prediction,
                row.weight,
            )
        )

fixed_accuracy = _pairwise_accuracy(fixed_records)
hier_accuracy = _pairwise_accuracy(hier_records)

print(
    "method\tmin_observed_margin_pt\tpair_accuracy\tweighted_pair_accuracy\tpairs"
)
for margin in MARGINS:
    accuracy, weighted, pairs = fixed_accuracy[margin]
    print(f"fixed_bbi\t{margin:.1f}\t{100*accuracy:.2f}%\t{100*weighted:.2f}%\t{pairs}")
    accuracy, weighted, pairs = hier_accuracy[margin]
    print(
        f"hierarchical_full\t{margin:.1f}\t{100*accuracy:.2f}%\t"
        f"{100*weighted:.2f}%\t{pairs}"
    )

print(
    "note=pairwise accuracy uses only model pairs held out on the same benchmark in the "
    "same family-CV fold; margin thresholds remove near-ties in normalized score points"
)
