# ruff: noqa: I001
from __future__ import annotations

import sys
from collections import Counter, defaultdict
from datetime import date

from better_bench.estimator import EstimatorConfig, _fit_state, _general_scale, _prepare
from better_bench.io import load_adoption, load_benchmarks, load_models, load_observations


if len(sys.argv) != 3:
    raise SystemExit("usage: python scripts/pairwise_overlap_report.py MODEL_A MODEL_B")

model_a, model_b = sys.argv[1:]
models = load_models("data/current")
benchmarks = load_benchmarks("data/current")
observations = load_observations("data/current")
adoption = load_adoption("data/current")
config = EstimatorConfig(jackknife_uncertainty=False)
(
    rows,
    model_ids,
    benchmark_ids,
    benchmark_by_id,
    _,
    _,
) = _prepare(models, benchmarks, observations, adoption, config, date(2026, 9, 4))
state, _, _ = _fit_state(rows, model_ids, benchmark_ids, config)
mean, scale = _general_scale(state, model_ids)

if model_a not in state.general or model_b not in state.general:
    raise SystemExit("both models must be retained by the estimator")

print(
    f"global_z\t{model_a}\t{(state.general[model_a] - mean) / scale:+.4f}\t"
    f"{model_b}\t{(state.general[model_b] - mean) / scale:+.4f}"
)

by_model: dict[str, list] = defaultdict(list)
for row in rows:
    if row.model_id in {model_a, model_b}:
        by_model[row.model_id].append(row)

for model_id in (model_a, model_b):
    counts = Counter(row.benchmark_id for row in by_model[model_id])
    duplicates = {key: value for key, value in counts.items() if value > 1}
    print(f"duplicates\t{model_id}\t{duplicates or '-'}")

print(
    "model\tbenchmark\tfamily\tscore\tweight\tintercept\tloading_pt_per_raw_g\t"
    "implied_raw_g\tnumerator_term\tdenominator_term"
)
for model_id in (model_a, model_b):
    for row in sorted(by_model[model_id], key=lambda item: (item.family_id, item.benchmark_id)):
        intercept = state.intercept[row.benchmark_id]
        loading = state.loading[row.benchmark_id]
        target = row.score_points / config.score_unit_points - intercept
        implied = target / loading if abs(loading) > 1e-12 else float("nan")
        numerator = row.weight * loading * target
        denominator = row.weight * loading * loading
        print(
            f"{model_id}\t{row.benchmark_id}\t{row.family_id}\t{row.score_points:.3f}\t"
            f"{row.weight:.4f}\t{config.score_unit_points * intercept:.3f}\t"
            f"{config.score_unit_points * loading:.3f}\t{implied:+.4f}\t"
            f"{numerator:+.5f}\t{denominator:.5f}"
        )

# Exact-benchmark overlap: aggregate repeated rows within a model first so duplicate
# source/protocol observations do not masquerade as additional matched benchmarks.
by_model_benchmark: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
for model_id in (model_a, model_b):
    for row in by_model[model_id]:
        by_model_benchmark[model_id][row.benchmark_id].append(row)

shared = sorted(
    set(by_model_benchmark[model_a]).intersection(by_model_benchmark[model_b])
)
print("shared_benchmark\tfamily\ta_score\tb_score\tdelta_score\tloading_pt_per_raw_g")
weighted_delta_num = 0.0
weighted_delta_den = 0.0
wins_a = wins_b = ties = 0
for benchmark_id in shared:
    a_rows = by_model_benchmark[model_a][benchmark_id]
    b_rows = by_model_benchmark[model_b][benchmark_id]
    a_weight = sum(row.weight for row in a_rows)
    b_weight = sum(row.weight for row in b_rows)
    a_score = sum(row.weight * row.score_points for row in a_rows) / a_weight
    b_score = sum(row.weight * row.score_points for row in b_rows) / b_weight
    delta = a_score - b_score
    if delta > 1e-9:
        wins_a += 1
    elif delta < -1e-9:
        wins_b += 1
    else:
        ties += 1
    loading = state.loading[benchmark_id]
    pair_weight = min(a_weight, b_weight)
    weighted_delta_num += pair_weight * loading * (delta / config.score_unit_points)
    weighted_delta_den += pair_weight * loading * loading
    family = benchmark_by_id[benchmark_id].family_id or benchmark_id
    print(
        f"{benchmark_id}\t{family}\t{a_score:.3f}\t{b_score:.3f}\t{delta:+.3f}\t"
        f"{config.score_unit_points * loading:.3f}"
    )

matched_raw_delta = weighted_delta_num / max(weighted_delta_den, 1e-12)
print(
    f"shared_summary\tbenchmarks={len(shared)}\t{model_a}_wins={wins_a}\t"
    f"{model_b}_wins={wins_b}\tties={ties}\tmatched_raw_g_delta={matched_raw_delta:+.4f}\t"
    f"matched_standardized_delta={matched_raw_delta / scale:+.4f}"
)
