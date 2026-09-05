# ruff: noqa: I001
from __future__ import annotations

from collections import defaultdict
from datetime import date

import numpy as np

from better_bench.consensus_ranking import (
    ConsensusRankingConfig,
    build_pair_preferences,
    fit_consensus_ranking,
)
from better_bench.estimator import EstimatorConfig, _balanced_folds, _fit_state, _prepare
from better_bench.io import load_adoption, load_benchmarks, load_models, load_observations

AS_OF = date(2026, 9, 4)
FOLDS = 5
MARGINS = (0.0, 1.0, 3.0, 5.0, 10.0)
CANDIDATES = tuple(
    ConsensusRankingConfig(
        margin_temperature_points=temperature,
        minimum_shared_families=minimum_shared,
        time_limit_seconds=20.0,
    )
    for temperature in (1.0, 3.0, 5.0, 10.0)
    for minimum_shared in (1, 2)
)


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


def _pair_counts(rows, scores):
    grouped = defaultdict(list)
    for row in rows:
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
                    weight = min(left.weight, right.weight)
                    result[margin][1] += 1
                    result[margin][3] += weight
                    if observed_delta * predicted_delta > 0:
                        result[margin][0] += 1
                        result[margin][2] += weight
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
rows, model_ids, benchmark_ids, _, _, _ = _prepare(
    models, benchmarks, observations, adoption, estimator_config, AS_OF
)
assignment = _balanced_folds(rows, FOLDS)

fixed_total = _empty_counts()
for fold in range(FOLDS):
    training = [row for row in rows if assignment[(row.model_id, row.family_id)] != fold]
    validation = [row for row in rows if assignment[(row.model_id, row.family_id)] == fold]
    state, _, _ = _fit_state(training, model_ids, benchmark_ids, estimator_config)
    _merge(fixed_total, _pair_counts(validation, _fixed_scores(state, model_ids)))
fixed_rates = _rates(fixed_total)
print("baseline\tmargin\tfixed_acc\tfixed_weighted\tpairs")
for margin in MARGINS:
    print(
        f"baseline\t{margin:.1f}\t{100*fixed_rates[margin][0]:.2f}%\t"
        f"{100*fixed_rates[margin][1]:.2f}%\t{fixed_rates[margin][2]}"
    )

results = []
for candidate in CANDIDATES:
    total = _empty_counts()
    statuses = []
    for fold in range(FOLDS):
        training = [row for row in rows if assignment[(row.model_id, row.family_id)] != fold]
        validation = [row for row in rows if assignment[(row.model_id, row.family_id)] == fold]
        state, _, _ = _fit_state(training, model_ids, benchmark_ids, estimator_config)
        consensus = fit_consensus_ranking(
            training,
            model_ids,
            state,
            score_unit_points=estimator_config.score_unit_points,
            config=candidate,
        )
        statuses.append(consensus.solver_status)
        _merge(total, _pair_counts(validation, consensus.scores))
    results.append((_objective(total), candidate, total, statuses))

results.sort(key=lambda item: item[0], reverse=True)
print("candidate\ttemperature\tmin_shared\tobjective\tall_acc\tweighted\tgt5\tstatus")
for objective, candidate, counts, statuses in results:
    rates = _rates(counts)
    print(
        f"candidate\t{candidate.margin_temperature_points:.1f}\t"
        f"{candidate.minimum_shared_families}\t{100*objective:.2f}%\t"
        f"{100*rates[0.0][0]:.2f}%\t{100*rates[0.0][1]:.2f}%\t"
        f"{100*rates[5.0][0]:.2f}%\t{','.join(str(value) for value in statuses)}"
    )

best = results[0][1]
full_state, _, _ = _fit_state(rows, model_ids, benchmark_ids, estimator_config)
consensus = fit_consensus_ranking(
    rows,
    model_ids,
    full_state,
    score_unit_points=estimator_config.score_unit_points,
    config=best,
)
print(
    f"deployment\ttemperature={best.margin_temperature_points:.1f}\t"
    f"min_shared={best.minimum_shared_families}\t"
    f"agreement={100*consensus.weighted_agreement/max(consensus.weighted_total,1e-12):.2f}%"
)
print("ranking\trank\tmodel")
for rank, model_id in enumerate(consensus.ranking, start=1):
    print(f"ranking\t{rank}\t{model_id}")

preferences = build_pair_preferences(
    rows,
    model_ids,
    full_state,
    score_unit_points=estimator_config.score_unit_points,
    config=best,
)
contradictions = 0
weighted_contradiction = 0.0
weighted_total = 0.0
for preference in preferences:
    if abs(preference.net_preference) <= 1e-12:
        continue
    predicted_left = consensus.rank[preference.left_model] < consensus.rank[preference.right_model]
    observed_left = preference.net_preference > 0
    weighted_total += abs(preference.net_preference)
    if predicted_left != observed_left:
        contradictions += 1
        weighted_contradiction += abs(preference.net_preference)
    if {preference.left_model, preference.right_model} == {"glm-5.3", "glm-5.3-flash"}:
        sign = 1.0 if preference.left_model == "glm-5.3" else -1.0
        print(
            f"glm_direct_preference\t{sign*preference.net_preference:+.6f}\t"
            f"families={preference.shared_families}\tbenchmarks={preference.shared_benchmarks}"
        )
print(
    f"contradictions\tcount={contradictions}/{len(preferences)}\t"
    f"weighted_mass={100*weighted_contradiction/max(weighted_total,1e-12):.2f}%"
)
print(
    f"glm_order\tmain={consensus.rank['glm-5.3']}\tflash={consensus.rank['glm-5.3-flash']}"
)
