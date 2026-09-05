# ruff: noqa: I001
from __future__ import annotations

from collections import defaultdict
from datetime import date

import numpy as np

from better_bench.bradley_terry import BradleyTerryConfig, fit_bradley_terry
from better_bench.estimator import EstimatorConfig, _fit_state, _prepare
from better_bench.io import load_adoption, load_benchmarks, load_models, load_observations
from better_bench.ranking_evidence import prepare_ranking_evidence

AS_OF = date(2026, 9, 4)
FOLDS = 5
FRONTIER_SIZE = 10
MARGINS = (0.0, 1.0, 3.0, 5.0, 10.0)
CANDIDATES = tuple(
    BradleyTerryConfig(
        margin_temperature_points=temperature,
        minimum_shared_families=minimum_shared,
        ridge=ridge,
        use_learned_discrimination=False,
    )
    for temperature in (1.0, 3.0, 5.0, 10.0)
    for minimum_shared in (1, 2)
    for ridge in (0.03, 0.10, 0.30, 1.0, 3.0)
)


def _family_folds(rows, folds):
    counts = defaultdict(int)
    for row in rows:
        counts[row.family_id] += 1
    loads = [0 for _ in range(folds)]
    assignment = {}
    for family_id, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        fold = min(range(folds), key=lambda idx: (loads[idx], idx))
        assignment[family_id] = fold
        loads[fold] += count
    return assignment


def _fixed_scores(state, model_ids):
    values = np.asarray([state.general[model_id] for model_id in model_ids], dtype=float)
    values -= float(values.mean())
    scale = float(values.std(ddof=0))
    if scale <= 1e-12:
        scale = 1.0
    values /= scale
    return {model_id: float(values[index]) for index, model_id in enumerate(model_ids)}


def _empty_counts():
    return {margin: [0, 0, 0.0, 0.0] for margin in MARGINS}


def _pair_counts(rows, scores, *, allowed_models=None):
    grouped = defaultdict(list)
    for row in rows:
        if allowed_models is not None and row.model_id not in allowed_models:
            continue
        grouped[row.benchmark_id].append(row)
    result = _empty_counts()
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
    return {
        margin: (
            correct / max(total, 1),
            weighted_correct / max(weighted_total, 1e-12),
            int(total),
        )
        for margin, (correct, total, weighted_correct, weighted_total) in counts.items()
    }


def _objective(counts):
    rates = _rates(counts)
    return float(np.mean([rates[margin][0] for margin in MARGINS]))


models = load_models("data/current")
benchmarks = load_benchmarks("data/current")
observations = load_observations("data/current")
adoption = load_adoption("data/current")
estimator_config = EstimatorConfig(jackknife_uncertainty=False)
calibration_rows, model_ids, benchmark_ids, _, _, _ = _prepare(
    models,
    benchmarks,
    observations,
    adoption,
    estimator_config,
    AS_OF,
)
panel = prepare_ranking_evidence(
    models,
    benchmarks,
    observations,
    adoption,
    rankable_model_ids=model_ids,
    as_of=AS_OF,
    minimum_models_per_benchmark=2,
)
rows = panel.observations
assignment = _family_folds(rows, FOLDS)

fixed_total = _empty_counts()
fixed_frontier_total = _empty_counts()
method_total = {candidate: _empty_counts() for candidate in CANDIDATES}
method_frontier = {candidate: _empty_counts() for candidate in CANDIDATES}
convergence = {candidate: [] for candidate in CANDIDATES}

for fold in range(FOLDS):
    heldout = {family for family, value in assignment.items() if value == fold}
    training = [row for row in rows if row.family_id not in heldout]
    validation = [row for row in rows if row.family_id in heldout]
    calibration_training = [row for row in calibration_rows if row.family_id not in heldout]
    state, _, _ = _fit_state(
        calibration_training,
        model_ids,
        benchmark_ids,
        estimator_config,
    )
    fixed = _fixed_scores(state, model_ids)
    frontier = set(
        sorted(model_ids, key=lambda model_id: fixed[model_id], reverse=True)[:FRONTIER_SIZE]
    )
    _merge(fixed_total, _pair_counts(validation, fixed))
    _merge(fixed_frontier_total, _pair_counts(validation, fixed, allowed_models=frontier))

    for candidate in CANDIDATES:
        result = fit_bradley_terry(
            training,
            model_ids,
            None,
            score_unit_points=estimator_config.score_unit_points,
            config=candidate,
        )
        convergence[candidate].append(result.converged)
        _merge(method_total[candidate], _pair_counts(validation, result.scores))
        _merge(
            method_frontier[candidate],
            _pair_counts(validation, result.scores, allowed_models=frontier),
        )

fixed_rates = _rates(fixed_total)
fixed_frontier_rates = _rates(fixed_frontier_total)
print(
    f"panel\tbenchmarks={len(panel.retained_benchmarks)}\tfamilies={len(panel.retained_families)}\t"
    f"observations={len(rows)}\tcalibration_benchmarks={len(benchmark_ids)}"
)
print("baseline\tmargin\tall\tweighted\tfrontier\tfrontier_weighted\tpairs")
for margin in MARGINS:
    print(
        f"baseline\t{margin:.1f}\t{100*fixed_rates[margin][0]:.2f}%\t"
        f"{100*fixed_rates[margin][1]:.2f}%\t"
        f"{100*fixed_frontier_rates[margin][0]:.2f}%\t"
        f"{100*fixed_frontier_rates[margin][1]:.2f}%\t{fixed_rates[margin][2]}"
    )

scored = []
for candidate in CANDIDATES:
    rates = _rates(method_total[candidate])
    frontier_rates = _rates(method_frontier[candidate])
    objective = _objective(method_total[candidate])
    scored.append((objective, candidate, rates, frontier_rates))
scored.sort(key=lambda item: item[0], reverse=True)

print(
    "candidate\ttemp\tmin_shared\tridge\tobjective\tall\tweighted\tgt5\t"
    "frontier\tfrontier_gt5\tconverged"
)
for objective, candidate, rates, frontier_rates in scored[:20]:
    print(
        f"candidate\t{candidate.margin_temperature_points:.1f}\t"
        f"{candidate.minimum_shared_families}\t{candidate.ridge:.2f}\t"
        f"{100*objective:.2f}%\t{100*rates[0.0][0]:.2f}%\t"
        f"{100*rates[0.0][1]:.2f}%\t{100*rates[5.0][0]:.2f}%\t"
        f"{100*frontier_rates[0.0][0]:.2f}%\t{100*frontier_rates[5.0][0]:.2f}%\t"
        f"{sum(convergence[candidate])}/{FOLDS}"
    )

best = scored[0][1]
full = fit_bradley_terry(
    rows,
    model_ids,
    None,
    score_unit_points=estimator_config.score_unit_points,
    config=best,
)
print(
    f"deployment\ttemp={best.margin_temperature_points:.1f}\t"
    f"min_shared={best.minimum_shared_families}\tridge={best.ridge:.2f}\t"
    f"converged={full.converged}\titerations={full.iterations}"
)
print("ranking\trank\tmodel\tbt_z")
for rank, model_id in enumerate(full.ranking, start=1):
    print(f"ranking\t{rank}\t{model_id}\t{full.scores[model_id]:+.4f}")
print(
    f"glm_order\tmain={full.rank['glm-5.3']}\tflash={full.rank['glm-5.3-flash']}\t"
    f"delta={full.scores['glm-5.3']-full.scores['glm-5.3-flash']:+.4f}"
)
print(
    f"gpt55_order\tgpt55={full.rank['gpt-5.5']}\tflash={full.rank['glm-5.3-flash']}\t"
    f"delta={full.scores['gpt-5.5']-full.scores['glm-5.3-flash']:+.4f}"
)
