# ruff: noqa: I001
from __future__ import annotations

from collections import defaultdict
from datetime import date

import numpy as np

from better_bench.consensus_ranking import ConsensusRankingConfig, fit_consensus_ranking
from better_bench.estimator import EstimatorConfig, _fit_state, _prepare
from better_bench.io import load_adoption, load_benchmarks, load_models, load_observations
from better_bench.ranking_evidence import prepare_ranking_evidence

AS_OF = date(2026, 9, 4)
FOLDS = 5
FRONTIER_SIZE = 10
MARGINS = (0.0, 1.0, 3.0, 5.0, 10.0)
CANDIDATES = tuple(
    ConsensusRankingConfig(
        margin_temperature_points=temperature,
        minimum_shared_families=minimum_shared,
        use_learned_discrimination=False,
        time_limit_seconds=20.0,
    )
    for temperature in (1.0, 3.0, 5.0, 10.0)
    for minimum_shared in (1, 2)
)


def _family_folds(rows, folds):
    """Assign whole benchmark families to folds, balancing held-out row counts."""
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
ranking_panel = prepare_ranking_evidence(
    models,
    benchmarks,
    observations,
    adoption,
    rankable_model_ids=model_ids,
    as_of=AS_OF,
    minimum_models_per_benchmark=2,
)
ranking_rows = ranking_panel.observations
assignment = _family_folds(ranking_rows, FOLDS)

print(
    f"panel\tmodels={len(model_ids)}\tbenchmarks={len(ranking_panel.retained_benchmarks)}\t"
    f"families={len(ranking_panel.retained_families)}\tobservations={len(ranking_rows)}\t"
    f"calibration_benchmarks={len(benchmark_ids)}\tcalibration_observations={len(calibration_rows)}"
)

fixed_total = _empty_counts()
fixed_frontier_total = _empty_counts()
consensus_totals = {candidate: _empty_counts() for candidate in CANDIDATES}
consensus_frontier_totals = {candidate: _empty_counts() for candidate in CANDIDATES}
solver_statuses = {candidate: [] for candidate in CANDIDATES}

for fold in range(FOLDS):
    heldout_families = {family for family, value in assignment.items() if value == fold}
    broad_training = [row for row in ranking_rows if row.family_id not in heldout_families]
    validation = [row for row in ranking_rows if row.family_id in heldout_families]
    calibration_training = [
        row for row in calibration_rows if row.family_id not in heldout_families
    ]
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
    _merge(
        fixed_frontier_total,
        _pair_counts(validation, fixed, allowed_models=frontier),
    )

    for candidate in CANDIDATES:
        consensus = fit_consensus_ranking(
            broad_training,
            model_ids,
            None,
            score_unit_points=estimator_config.score_unit_points,
            config=candidate,
        )
        solver_statuses[candidate].append(consensus.solver_status)
        _merge(consensus_totals[candidate], _pair_counts(validation, consensus.scores))
        _merge(
            consensus_frontier_totals[candidate],
            _pair_counts(validation, consensus.scores, allowed_models=frontier),
        )

fixed_rates = _rates(fixed_total)
fixed_frontier_rates = _rates(fixed_frontier_total)
print("baseline\tmargin\tfixed\tfixed_weighted\tfrontier\tfrontier_weighted\tpairs")
for margin in MARGINS:
    print(
        f"baseline\t{margin:.1f}\t{100*fixed_rates[margin][0]:.2f}%\t"
        f"{100*fixed_rates[margin][1]:.2f}%\t"
        f"{100*fixed_frontier_rates[margin][0]:.2f}%\t"
        f"{100*fixed_frontier_rates[margin][1]:.2f}%\t{fixed_rates[margin][2]}"
    )

scored = []
print(
    "candidate\ttemperature\tmin_shared\tobjective\tall\tweighted\tgt5\t"
    "frontier\tfrontier_gt5\tstatus"
)
for candidate in CANDIDATES:
    rates = _rates(consensus_totals[candidate])
    frontier_rates = _rates(consensus_frontier_totals[candidate])
    objective = _objective(consensus_totals[candidate])
    scored.append((objective, candidate))
    print(
        f"candidate\t{candidate.margin_temperature_points:.1f}\t"
        f"{candidate.minimum_shared_families}\t{100*objective:.2f}%\t"
        f"{100*rates[0.0][0]:.2f}%\t{100*rates[0.0][1]:.2f}%\t"
        f"{100*rates[5.0][0]:.2f}%\t{100*frontier_rates[0.0][0]:.2f}%\t"
        f"{100*frontier_rates[5.0][0]:.2f}%\t"
        f"{','.join(str(value) for value in solver_statuses[candidate])}"
    )

scored.sort(key=lambda item: item[0], reverse=True)
best = scored[0][1]
full_consensus = fit_consensus_ranking(
    ranking_rows,
    model_ids,
    None,
    score_unit_points=estimator_config.score_unit_points,
    config=best,
)
print(
    f"deployment\ttemperature={best.margin_temperature_points:.1f}\t"
    f"min_shared={best.minimum_shared_families}\t"
    f"agreement={100*full_consensus.weighted_agreement/max(full_consensus.weighted_total,1e-12):.2f}%"
)
print("ranking\trank\tmodel")
for rank, model_id in enumerate(full_consensus.ranking, start=1):
    print(f"ranking\t{rank}\t{model_id}")
print(
    f"glm_order\tmain={full_consensus.rank['glm-5.3']}\t"
    f"flash={full_consensus.rank['glm-5.3-flash']}"
)
print(
    f"gpt55_order\tgpt55={full_consensus.rank['gpt-5.5']}\t"
    f"flash={full_consensus.rank['glm-5.3-flash']}"
)
