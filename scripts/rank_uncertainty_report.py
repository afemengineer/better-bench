# ruff: noqa: I001
from __future__ import annotations

import math
from collections import defaultdict
from datetime import date
from itertools import combinations

import numpy as np
from scipy.stats import norm

from better_bench.estimator import (
    EstimatorConfig,
    _fit_state,
    _general_scale,
    _prepare,
    fit_estimator,
)
from better_bench.io import load_adoption, load_benchmarks, load_models, load_observations


AS_OF = date(2026, 9, 4)
ALPHA = 0.05

models = load_models("data/current")
benchmarks = load_benchmarks("data/current")
observations = load_observations("data/current")
adoption = load_adoption("data/current")
config = EstimatorConfig(jackknife_uncertainty=False)

estimator = fit_estimator(
    models,
    benchmarks,
    observations,
    adoption,
    config=config,
    as_of=AS_OF,
)
rows, model_ids, benchmark_ids, _, _, _ = _prepare(
    models,
    benchmarks,
    observations,
    adoption,
    config,
    AS_OF,
)

base = {row.model_id: row for row in estimator.models}
families = sorted({row.family_id for row in rows})
jackknife_draws: dict[str, list[float]] = defaultdict(list)
for omitted in families:
    state, _, _ = _fit_state(
        rows,
        model_ids,
        benchmark_ids,
        config,
        family_multipliers={omitted: 0.0},
    )
    mean, scale = _general_scale(state, model_ids)
    for model_id in model_ids:
        jackknife_draws[model_id].append((state.general[model_id] - mean) / scale)


pair_rows = []
family_count = len(families)
for model_a, model_b in combinations(model_ids, 2):
    delta = base[model_a].general_z - base[model_b].general_z
    draw_delta = np.asarray(jackknife_draws[model_a], dtype=float) - np.asarray(
        jackknife_draws[model_b], dtype=float
    )
    draw_mean = float(draw_delta.mean())
    family_variance = (
        (family_count - 1)
        / family_count
        * float(np.square(draw_delta - draw_mean).sum())
    )
    # Conditional terms are treated as independent here. This is conservative when
    # benchmark-level noise is positively correlated across models and is explicitly
    # kept separate from the joint family-jackknife covariance above.
    conditional_variance = (
        base[model_a].conditional_se**2 + base[model_b].conditional_se**2
    )
    pair_se = math.sqrt(max(family_variance + conditional_variance, 1e-12))
    z_value = delta / pair_se
    p_value = float(2.0 * norm.sf(abs(z_value)))
    pair_rows.append(
        {
            "model_a": model_a,
            "model_b": model_b,
            "delta": delta,
            "se": pair_se,
            "z": z_value,
            "p": p_value,
            "p_holm": 1.0,
        }
    )

# Holm step-down adjustment across all unordered model-pair contrasts.
order = sorted(range(len(pair_rows)), key=lambda index: pair_rows[index]["p"])
running = 0.0
m = len(pair_rows)
for rank, index in enumerate(order):
    adjusted = min(1.0, (m - rank) * pair_rows[index]["p"])
    running = max(running, adjusted)
    pair_rows[index]["p_holm"] = running

significant_better: dict[str, set[str]] = defaultdict(set)
for row in pair_rows:
    if row["p_holm"] >= ALPHA:
        continue
    if row["delta"] > 0:
        significant_better[row["model_a"]].add(row["model_b"])
    elif row["delta"] < 0:
        significant_better[row["model_b"]].add(row["model_a"])

ranked = sorted(base.values(), key=lambda row: row.general_z, reverse=True)
nominal_rank = {row.model_id: rank for rank, row in enumerate(ranked, start=1)}

print(
    "method=joint_family_jackknife_plus_conditional_normal_approx "
    f"families={family_count} pairs={len(pair_rows)} alpha={ALPHA:.2f} correction=holm"
)
print(
    "note=rank intervals are inferential diagnostics for the current estimator, not "
    "a claim that benchmark-family observations are iid"
)
print("rank\tmodel\tgeneral_z\trank_interval\tdefinitely_better\tdefinitely_worse")
for row in ranked:
    better_than = significant_better.get(row.model_id, set())
    worse_than = {
        other
        for other, beaten in significant_better.items()
        if row.model_id in beaten
    }
    best_rank = 1 + len(worse_than)
    worst_rank = len(model_ids) - len(better_than)
    print(
        f"{nominal_rank[row.model_id]}\t{row.model_id}\t{row.general_z:+.4f}\t"
        f"[{best_rank},{worst_rank}]\t{len(worse_than)}\t{len(better_than)}"
    )

print("selected_pair\tdelta_z\tse_z\tz\tp_raw\tp_holm\tresolved")
selected = {
    frozenset(("glm-5.3", "glm-5.3-flash")),
    frozenset(("glm-5.3-flash", "gpt-5.5")),
    frozenset(("gpt-6-astra", "claude-fable-5.1")),
}
for row in pair_rows:
    if frozenset((row["model_a"], row["model_b"])) not in selected:
        continue
    label = f"{row['model_a']} - {row['model_b']}"
    print(
        f"{label}\t{row['delta']:+.4f}\t{row['se']:.4f}\t{row['z']:+.3f}\t"
        f"{row['p']:.4g}\t{row['p_holm']:.4g}\t"
        f"{'yes' if row['p_holm'] < ALPHA else 'no'}"
    )
