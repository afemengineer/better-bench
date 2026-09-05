# ruff: noqa: I001
from __future__ import annotations

from collections import defaultdict
from datetime import date

import numpy as np
from scipy.stats import spearmanr

from better_bench.consensus_ranking import (
    ConsensusRankingConfig,
    build_pair_preferences,
    fit_consensus_ranking,
)
from better_bench.estimator import EstimatorConfig, _fit_state, _prepare
from better_bench.io import load_adoption, load_benchmarks, load_models, load_observations
from better_bench.ranking_evidence import prepare_ranking_evidence

AS_OF = date(2026, 9, 4)
OUTER_FOLDS = 5
INNER_FOLDS = 4
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


def _fit(rows, model_ids, candidate):
    return fit_consensus_ranking(
        rows,
        model_ids,
        None,
        score_unit_points=10.0,
        config=candidate,
    )


def _select_candidate(training_rows, model_ids):
    inner_assignment = _family_folds(training_rows, INNER_FOLDS)
    scored = []
    for candidate in CANDIDATES:
        counts = _empty_counts()
        for fold in range(INNER_FOLDS):
            heldout = {family for family, value in inner_assignment.items() if value == fold}
            inner_train = [row for row in training_rows if row.family_id not in heldout]
            inner_val = [row for row in training_rows if row.family_id in heldout]
            result = _fit(inner_train, model_ids, candidate)
            _merge(counts, _pair_counts(inner_val, result.scores))
        scored.append((_objective(counts), candidate))
    scored.sort(
        key=lambda item: (
            item[0],
            -item[1].minimum_shared_families,
            item[1].margin_temperature_points,
        ),
        reverse=True,
    )
    return scored[0]


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
outer_assignment = _family_folds(rows, OUTER_FOLDS)

fixed_total = _empty_counts()
consensus_total = _empty_counts()
fixed_frontier_total = _empty_counts()
consensus_frontier_total = _empty_counts()

print(
    "outer_fold\tselected_temp\tselected_min_shared\tinner_objective\t"
    "fixed_acc\tconsensus_acc\tfrontier_fixed\tfrontier_consensus"
)
for outer_fold in range(OUTER_FOLDS):
    outer_heldout = {
        family for family, value in outer_assignment.items() if value == outer_fold
    }
    training = [row for row in rows if row.family_id not in outer_heldout]
    validation = [row for row in rows if row.family_id in outer_heldout]
    inner_objective, selected = _select_candidate(training, model_ids)
    consensus = _fit(training, model_ids, selected)

    calibration_training = [
        row for row in calibration_rows if row.family_id not in outer_heldout
    ]
    fixed_state, _, _ = _fit_state(
        calibration_training,
        model_ids,
        benchmark_ids,
        estimator_config,
    )
    fixed = _fixed_scores(fixed_state, model_ids)
    frontier = set(
        sorted(model_ids, key=lambda model_id: fixed[model_id], reverse=True)[:FRONTIER_SIZE]
    )

    fixed_counts = _pair_counts(validation, fixed)
    consensus_counts = _pair_counts(validation, consensus.scores)
    fixed_frontier = _pair_counts(validation, fixed, allowed_models=frontier)
    consensus_frontier = _pair_counts(
        validation,
        consensus.scores,
        allowed_models=frontier,
    )
    _merge(fixed_total, fixed_counts)
    _merge(consensus_total, consensus_counts)
    _merge(fixed_frontier_total, fixed_frontier)
    _merge(consensus_frontier_total, consensus_frontier)

    fixed_rate = _rates(fixed_counts)[0.0][0]
    consensus_rate = _rates(consensus_counts)[0.0][0]
    fixed_frontier_rate = _rates(fixed_frontier)[0.0][0]
    consensus_frontier_rate = _rates(consensus_frontier)[0.0][0]
    print(
        f"{outer_fold}\t{selected.margin_temperature_points:.1f}\t"
        f"{selected.minimum_shared_families}\t{100*inner_objective:.2f}%\t"
        f"{100*fixed_rate:.2f}%\t{100*consensus_rate:.2f}%\t"
        f"{100*fixed_frontier_rate:.2f}%\t{100*consensus_frontier_rate:.2f}%"
    )

fixed_rates = _rates(fixed_total)
consensus_rates = _rates(consensus_total)
fixed_frontier_rates = _rates(fixed_frontier_total)
consensus_frontier_rates = _rates(consensus_frontier_total)
print("nested_summary\tmargin\tfixed\tconsensus\tfixed_weighted\tconsensus_weighted\tpairs")
for margin in MARGINS:
    print(
        f"nested_summary\t{margin:.1f}\t{100*fixed_rates[margin][0]:.2f}%\t"
        f"{100*consensus_rates[margin][0]:.2f}%\t"
        f"{100*fixed_rates[margin][1]:.2f}%\t"
        f"{100*consensus_rates[margin][1]:.2f}%\t{fixed_rates[margin][2]}"
    )
print("frontier_summary\tmargin\tfixed\tconsensus\tfixed_weighted\tconsensus_weighted\tpairs")
for margin in MARGINS:
    print(
        f"frontier_summary\t{margin:.1f}\t{100*fixed_frontier_rates[margin][0]:.2f}%\t"
        f"{100*consensus_frontier_rates[margin][0]:.2f}%\t"
        f"{100*fixed_frontier_rates[margin][1]:.2f}%\t"
        f"{100*consensus_frontier_rates[margin][1]:.2f}%\t"
        f"{fixed_frontier_rates[margin][2]}"
    )

# Deployment hyperparameters are selected from the complete five-fold family-CV panel
# only after nested CV has estimated selection-time generalization.
deployment_scored = []
for candidate in CANDIDATES:
    counts = _empty_counts()
    for fold in range(OUTER_FOLDS):
        heldout = {family for family, value in outer_assignment.items() if value == fold}
        train = [row for row in rows if row.family_id not in heldout]
        validation = [row for row in rows if row.family_id in heldout]
        result = _fit(train, model_ids, candidate)
        _merge(counts, _pair_counts(validation, result.scores))
    deployment_scored.append((_objective(counts), candidate))
deployment_scored.sort(
    key=lambda item: (
        item[0],
        -item[1].minimum_shared_families,
        item[1].margin_temperature_points,
    ),
    reverse=True,
)
deployment_objective, deployment = deployment_scored[0]
full = _fit(rows, model_ids, deployment)
print(
    f"deployment\ttemp={deployment.margin_temperature_points:.1f}\t"
    f"min_shared={deployment.minimum_shared_families}\t"
    f"cv_objective={100*deployment_objective:.2f}%\t"
    f"agreement={100*full.weighted_agreement/max(full.weighted_total,1e-12):.2f}%"
)

preferences = build_pair_preferences(
    rows,
    model_ids,
    None,
    score_unit_points=10.0,
    config=deployment,
)
contradictions = 0
weighted_contradiction = 0.0
weighted_total = 0.0
for preference in preferences:
    magnitude = abs(preference.net_preference)
    if magnitude <= 1e-12:
        continue
    predicted_left = full.rank[preference.left_model] < full.rank[preference.right_model]
    observed_left = preference.net_preference > 0
    weighted_total += magnitude
    if predicted_left != observed_left:
        contradictions += 1
        weighted_contradiction += magnitude
    pair = {preference.left_model, preference.right_model}
    if pair in (
        {"glm-5.3", "glm-5.3-flash"},
        {"gpt-5.5", "glm-5.3-flash"},
    ):
        label = "glm" if "glm-5.3" in pair else "gpt55"
        reference = "glm-5.3" if label == "glm" else "gpt-5.5"
        sign = 1.0 if preference.left_model == reference else -1.0
        print(
            f"direct_preference\t{label}\t{sign*preference.net_preference:+.6f}\t"
            f"information={preference.total_information:.6f}\t"
            f"families={preference.shared_families}\tbenchmarks={preference.shared_benchmarks}"
        )
print(
    f"direct_contradictions\tcount={contradictions}/{len(preferences)}\t"
    f"weighted_mass={100*weighted_contradiction/max(weighted_total,1e-12):.2f}%"
)

# Leave-one-family-out stability is intentionally performed with the fixed deployment
# configuration: it measures sensitivity of the final method, not another tuning loop.
full_rank_vector = np.asarray([full.rank[model_id] for model_id in model_ids], dtype=float)
full_top10 = set(full.ranking[:10])
spearman_values = []
top10_overlaps = []
glm_main_over_flash = 0
gpt55_over_flash = 0
rank_samples = defaultdict(list)
for omitted in sorted({row.family_id for row in rows}):
    reduced = [row for row in rows if row.family_id != omitted]
    result = _fit(reduced, model_ids, deployment)
    rank_vector = np.asarray([result.rank[model_id] for model_id in model_ids], dtype=float)
    rho = float(spearmanr(full_rank_vector, rank_vector).statistic)
    spearman_values.append(rho)
    top10_overlaps.append(len(full_top10 & set(result.ranking[:10])) / 10.0)
    glm_main_over_flash += int(result.rank["glm-5.3"] < result.rank["glm-5.3-flash"])
    gpt55_over_flash += int(result.rank["gpt-5.5"] < result.rank["glm-5.3-flash"])
    for model_id in model_ids:
        rank_samples[model_id].append(result.rank[model_id])

print(
    f"family_stability\tomissions={len(spearman_values)}\t"
    f"median_spearman={np.median(spearman_values):.4f}\t"
    f"min_spearman={np.min(spearman_values):.4f}\t"
    f"median_top10={100*np.median(top10_overlaps):.1f}%\t"
    f"glm_main_over_flash={glm_main_over_flash}/{len(spearman_values)}\t"
    f"gpt55_over_flash={gpt55_over_flash}/{len(spearman_values)}"
)
print("ranking\trank\tmodel\tloo_min_rank\tloo_max_rank")
for rank, model_id in enumerate(full.ranking, start=1):
    samples = rank_samples[model_id]
    print(f"ranking\t{rank}\t{model_id}\t{min(samples)}\t{max(samples)}")
