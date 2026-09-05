from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from .consensus_ranking import ConsensusRankingConfig, fit_consensus_ranking
from .estimator import EstimatorResult
from .ranking_evidence import prepare_ranking_evidence
from .schema import (
    BenchmarkAdoptionSnapshot,
    BenchmarkDefinition,
    BenchmarkObservation,
    ModelDefinition,
)


RANKING_METHOD = "broad-kemeny-v1"
DEPLOYMENT_CONFIG = ConsensusRankingConfig(
    margin_temperature_points=10.0,
    minimum_shared_families=1,
    use_learned_discrimination=False,
    time_limit_seconds=30.0,
)


@dataclass(frozen=True)
class GeneralIntelligenceRank:
    model_id: str
    rank: int
    rank_low: int
    rank_high: int
    calibration_z: float
    ranking_family_count: int


@dataclass(frozen=True)
class GeneralIntelligenceRanking:
    models: list[GeneralIntelligenceRank]
    method: str
    benchmark_count: int
    family_count: int
    weighted_agreement: float

    @property
    def by_model(self) -> dict[str, GeneralIntelligenceRank]:
        return {row.model_id: row for row in self.models}


def build_general_intelligence_ranking(
    models: list[ModelDefinition],
    benchmarks: list[BenchmarkDefinition],
    observations: list[BenchmarkObservation],
    estimator: EstimatorResult,
    adoption: list[BenchmarkAdoptionSnapshot] | None = None,
    *,
    as_of: date | None = None,
    include_rank_bands: bool = True,
) -> GeneralIntelligenceRanking:
    """Build the official Better Bench general-intelligence ordering.

    The fixed one-factor estimator remains the conservative admission/calibration model.
    Final ordering is deliberately a separate statistical object: a weighted Kemeny
    consensus over every revision-safe benchmark that compares at least two already-
    rankable models. Missing cells provide no positive evidence, benchmark difficulty
    cancels within each comparison, and learned leaderboard discrimination is excluded
    from the ranking weights to avoid feeding the old ordering back into the correction.

    Rank bands are leave-one-benchmark-family-out sensitivity intervals. They should be
    interpreted as robustness bands, not calibrated frequentist confidence intervals.
    """

    as_of = as_of or date.today()
    adoption = adoption or []
    model_ids = list(estimator.retained_models)
    calibration_z = {row.model_id: row.general_z for row in estimator.models}

    panel = prepare_ranking_evidence(
        models,
        benchmarks,
        observations,
        adoption,
        rankable_model_ids=model_ids,
        as_of=as_of,
        minimum_models_per_benchmark=2,
    )
    full = fit_consensus_ranking(
        panel.observations,
        model_ids,
        None,
        score_unit_points=10.0,
        config=DEPLOYMENT_CONFIG,
    )

    rank_samples: dict[str, list[int]] = {
        model_id: [full.rank[model_id]] for model_id in model_ids
    }
    if include_rank_bands:
        for omitted_family in panel.retained_families:
            reduced = [
                row for row in panel.observations if row.family_id != omitted_family
            ]
            result = fit_consensus_ranking(
                reduced,
                model_ids,
                None,
                score_unit_points=10.0,
                config=DEPLOYMENT_CONFIG,
            )
            for model_id in model_ids:
                rank_samples[model_id].append(result.rank[model_id])

    rows = [
        GeneralIntelligenceRank(
            model_id=model_id,
            rank=full.rank[model_id],
            rank_low=min(rank_samples[model_id]),
            rank_high=max(rank_samples[model_id]),
            calibration_z=float(calibration_z[model_id]),
            ranking_family_count=len(panel.retained_families),
        )
        for model_id in model_ids
    ]
    rows.sort(key=lambda row: row.rank)
    return GeneralIntelligenceRanking(
        models=rows,
        method=RANKING_METHOD,
        benchmark_count=len(panel.retained_benchmarks),
        family_count=len(panel.retained_families),
        weighted_agreement=(
            full.weighted_agreement / full.weighted_total
            if full.weighted_total > 1e-12
            else 0.0
        ),
    )
