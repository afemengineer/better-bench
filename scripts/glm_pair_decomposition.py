# ruff: noqa: I001
from __future__ import annotations

from collections import defaultdict
from datetime import date

import numpy as np

from better_bench.estimator import EstimatorConfig, _fit_state, _prepare
from better_bench.io import load_adoption, load_benchmarks, load_models, load_observations
from better_bench.pairwise_ranking import PairwiseRankingConfig, build_pairwise_edges

AS_OF = date(2026, 9, 4)
LEFT = "glm-5.3"
RIGHT = "glm-5.3-flash"

models = load_models("data/current")
benchmarks = load_benchmarks("data/current")
observations = load_observations("data/current")
adoption = load_adoption("data/current")
config = EstimatorConfig(jackknife_uncertainty=False)
rows, model_ids, benchmark_ids, benchmark_by_id, _, _ = _prepare(
    models, benchmarks, observations, adoption, config, AS_OF
)
state, _, _ = _fit_state(rows, model_ids, benchmark_ids, config)

values = np.asarray([state.general[model_id] for model_id in model_ids], dtype=float)
general_scale = float(values.std(ddof=0))

cells = {(row.model_id, row.benchmark_id): row for row in rows}
benchmark_total = defaultdict(float)
for row in rows:
    benchmark_total[row.benchmark_id] += row.weight

shared = sorted(
    {row.benchmark_id for row in rows if row.model_id == LEFT}
    & {row.benchmark_id for row in rows if row.model_id == RIGHT}
)

families = defaultdict(list)
print("benchmark\tfamily\tdelta_pt\tloading_pt_per_z\tleft_w\tright_w\ttotal_w\tpair_info\tdelta_z")
for benchmark_id in shared:
    left = cells[(LEFT, benchmark_id)]
    right = cells[(RIGHT, benchmark_id)]
    loading_pt = config.score_unit_points * state.loading[benchmark_id] * general_scale
    if loading_pt <= 1e-8:
        continue
    delta_pt = left.score_points - right.score_points
    delta_z = delta_pt / loading_pt
    info_multiplier = (loading_pt / 10.0) ** 2
    pair_info = left.weight * right.weight / benchmark_total[benchmark_id] * info_multiplier
    family_id = benchmark_by_id[benchmark_id].family_id or benchmark_id
    families[family_id].append((delta_z, pair_info, benchmark_id))
    print(
        f"{benchmark_id}\t{family_id}\t{delta_pt:+.3f}\t{loading_pt:.3f}\t"
        f"{left.weight:.4f}\t{right.weight:.4f}\t{benchmark_total[benchmark_id]:.4f}\t"
        f"{pair_info:.6f}\t{delta_z:+.4f}"
    )

print("family\tdelta_z\tinformation\tbenchmarks")
for family_id, values in sorted(families.items()):
    information = np.asarray([item[1] for item in values], dtype=float)
    deltas = np.asarray([item[0] for item in values], dtype=float)
    total = float(information.sum())
    delta = float(np.average(deltas, weights=information)) if total > 0 else 0.0
    names = ",".join(item[2] for item in values)
    print(f"{family_id}\t{delta:+.4f}\t{total:.6f}\t{names}")

edges = build_pairwise_edges(
    rows,
    model_ids,
    state,
    score_unit_points=config.score_unit_points,
    config=PairwiseRankingConfig(minimum_shared_families=1, ridge=1.0, prior_mix=0.0),
)
for edge in edges:
    if {edge.left_model, edge.right_model} == {LEFT, RIGHT}:
        sign = 1.0 if edge.left_model == LEFT else -1.0
        print(
            f"DIRECT_EDGE\tdelta_z={sign*edge.delta_z:+.4f}\tweight={edge.weight:.6f}\t"
            f"shared_families={edge.shared_families}\tshared_benchmarks={edge.shared_benchmarks}"
        )
        break
