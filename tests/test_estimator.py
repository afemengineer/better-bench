from __future__ import annotations

from datetime import date

from better_bench.estimator import EstimatorConfig, cross_validate_estimator, fit_estimator
from better_bench.schema import (
    BenchmarkDefinition,
    BenchmarkObservation,
    Capability,
    ModelDefinition,
    SourceGrade,
)


def _problem() -> tuple[
    list[ModelDefinition], list[BenchmarkDefinition], list[BenchmarkObservation]
]:
    models = [
        ModelDefinition(
            id=f"m{index}",
            name=f"Model {index}",
            organization=f"org-{index % 4}",
            released_at=date(2026, 1, 1),
        )
        for index in range(12)
    ]
    specs = [
        ("reason-a", Capability.FLUID_REASONING, 42.0, 1.30),
        ("reason-b", Capability.FLUID_REASONING, 69.0, 0.35),
        ("software-a", Capability.SOFTWARE_ENGINEERING, 34.0, 1.10),
        ("software-b", Capability.SOFTWARE_ENGINEERING, 57.0, 0.80),
        ("visual-a", Capability.VISUAL_INTELLIGENCE, 48.0, 0.95),
        ("visual-b", Capability.VISUAL_INTELLIGENCE, 61.0, 0.55),
        ("quant-a", Capability.QUANTITATIVE_REASONING, 51.0, 1.20),
        ("social-a", Capability.SOCIAL_PRAGMATIC, 46.0, 0.70),
    ]
    benchmarks = [
        BenchmarkDefinition(
            id=benchmark_id,
            name=benchmark_id,
            family_id=f"family-{index}",
            published_at=date(2025, 1, 1),
            protocol_quality=0.95,
            reliability=0.95,
            contamination_resistance=0.9,
            capability_loadings={capability: 1.0},
        )
        for index, (benchmark_id, capability, _, _) in enumerate(specs)
    ]

    observations: list[BenchmarkObservation] = []
    for model_index, model in enumerate(models):
        ability = -1.6 + model_index * (3.2 / 11)
        for benchmark_id, capability, intercept, loading in specs:
            domain_bonus = 0.0
            if model.id == "m8" and capability == Capability.SOFTWARE_ENGINEERING:
                domain_bonus = 5.0
            if model.id == "m3" and capability == Capability.VISUAL_INTELLIGENCE:
                domain_bonus = 4.0
            score = intercept + 8.0 * loading * ability + domain_bonus
            observations.append(
                BenchmarkObservation(
                    model_id=model.id,
                    benchmark_id=benchmark_id,
                    score=max(0.0, min(100.0, score)),
                    evaluated_at=date(2026, 7, 1),
                    source_grade=SourceGrade.A,
                    source_url="https://independent-evaluator.test/results",
                )
            )
    return models, benchmarks, observations


def test_fixed_scale_estimator_recovers_general_order_and_discrimination() -> None:
    models, benchmarks, observations = _problem()
    result = fit_estimator(
        models,
        benchmarks,
        observations,
        config=EstimatorConfig(jackknife_uncertainty=True),
        as_of=date(2026, 9, 4),
    )

    by_model = {row.model_id: row for row in result.models}
    by_benchmark = {row.benchmark_id: row for row in result.benchmarks}
    assert result.converged
    assert by_model["m11"].general_z > by_model["m6"].general_z > by_model["m0"].general_z
    assert by_model["m8"].domain_residual_points["software_engineering"] > 0
    assert by_model["m3"].domain_residual_points["visual_intelligence"] > 0
    assert (
        by_benchmark["reason-a"].general_loading_points_per_z
        > by_benchmark["reason-b"].general_loading_points_per_z
    )
    assert by_model["m11"].family_jackknife_se >= 0
    assert by_model["m11"].ci_low < by_model["m11"].general_z < by_model["m11"].ci_high


def test_fixed_scale_family_holdout_predicts_raw_scores() -> None:
    models, benchmarks, observations = _problem()
    result = cross_validate_estimator(
        models,
        benchmarks,
        observations,
        config=EstimatorConfig(jackknife_uncertainty=False),
        folds=4,
        as_of=date(2026, 9, 4),
    )

    assert result.heldout_observations > 0
    assert result.model_rmse_points < result.benchmark_only_rmse_points
    assert result.relative_rmse_improvement > 0.20
    assert result.residual_r2 > 0.30
    assert result.residual_spearman > 0.50
    assert result.model_rmse_points < 10.0
