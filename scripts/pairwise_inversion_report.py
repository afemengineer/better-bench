# ruff: noqa: I001
from __future__ import annotations

from collections import defaultdict
from datetime import date
from itertools import combinations

from better_bench.estimator import EstimatorConfig, _fit_state, _general_scale, _prepare
from better_bench.io import load_adoption, load_benchmarks, load_models, load_observations


MIN_SHARED_BENCHMARKS = 5
MIN_SHARED_FAMILIES = 4
AS_OF = date(2026, 9, 4)

models = load_models("data/current")
benchmarks = load_benchmarks("data/current")
observations = load_observations("data/current")
adoption = load_adoption("data/current")
config = EstimatorConfig(jackknife_uncertainty=False)
rows, model_ids, benchmark_ids, benchmark_by_id, _, _ = _prepare(
    models,
    benchmarks,
    observations,
    adoption,
    config,
    AS_OF,
)
state, _, _ = _fit_state(rows, model_ids, benchmark_ids, config)
mean, scale = _general_scale(state, model_ids)

by_model_benchmark: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
for row in rows:
    by_model_benchmark[row.model_id][row.benchmark_id].append(row)


def aggregate(model_id: str, benchmark_id: str) -> tuple[float, float]:
    group = by_model_benchmark[model_id][benchmark_id]
    total_weight = sum(row.weight for row in group)
    score = sum(row.weight * row.score_points for row in group) / total_weight
    return score, total_weight


records = []
for model_a, model_b in combinations(model_ids, 2):
    shared = sorted(
        set(by_model_benchmark[model_a]).intersection(by_model_benchmark[model_b])
    )
    families = {
        benchmark_by_id[benchmark_id].family_id or benchmark_id
        for benchmark_id in shared
    }
    if len(shared) < MIN_SHARED_BENCHMARKS or len(families) < MIN_SHARED_FAMILIES:
        continue

    numerator = 0.0
    denominator = 0.0
    wins_a = wins_b = ties = 0
    for benchmark_id in shared:
        score_a, weight_a = aggregate(model_a, benchmark_id)
        score_b, weight_b = aggregate(model_b, benchmark_id)
        delta = score_a - score_b
        if delta > 1e-9:
            wins_a += 1
        elif delta < -1e-9:
            wins_b += 1
        else:
            ties += 1
        pair_weight = min(weight_a, weight_b)
        loading = state.loading[benchmark_id]
        numerator += pair_weight * loading * (delta / config.score_unit_points)
        denominator += pair_weight * loading * loading

    matched_delta = numerator / max(denominator, 1e-12) / scale
    global_delta = (
        state.general[model_a] - state.general[model_b]
    ) / scale
    if global_delta == 0.0 or matched_delta == 0.0 or global_delta * matched_delta > 0:
        continue
    records.append(
        (
            abs(global_delta - matched_delta),
            model_a,
            model_b,
            global_delta,
            matched_delta,
            len(shared),
            len(families),
            wins_a,
            wins_b,
            ties,
        )
    )

records.sort(reverse=True)
print(
    "severity\tmodel_a\tmodel_b\tglobal_delta_z\tmatched_delta_z\t"
    "shared_bench\tshared_fam\ta_wins\tb_wins\tties"
)
for record in records[:30]:
    (
        severity,
        model_a,
        model_b,
        global_delta,
        matched_delta,
        shared_count,
        family_count,
        wins_a,
        wins_b,
        ties,
    ) = record
    print(
        f"{severity:.4f}\t{model_a}\t{model_b}\t{global_delta:+.4f}\t"
        f"{matched_delta:+.4f}\t{shared_count}\t{family_count}\t"
        f"{wins_a}\t{wins_b}\t{ties}"
    )

print(f"inversions={len(records)}")
