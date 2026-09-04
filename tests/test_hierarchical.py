from __future__ import annotations

from datetime import date

from better_bench.hierarchical import HierarchicalConfig, fit_hierarchical
from better_bench.schema import (
    BenchmarkDefinition,
    BenchmarkObservation,
    Capability,
    ModelDefinition,
    SourceGrade,
)


def _synthetic_problem() -> tuple[
    list[ModelDefinition], list[BenchmarkDefinition], list[BenchmarkObservation]
]:
    models = [
        ModelDefinition(
            id=f"m{index}",
            name=f"Model {index}",
            organization=f"org-{index % 5}",
            released_at=date(2026, 6, 1),
        )
        for index in range(10)
    ]
    benchmark_specs = [
        ("fluid-a", Capability.FLUID_REASONING, False),
        ("fluid-b", Capability.FLUID_REASONING, True),
        ("software-a", Capability.SOFTWARE_ENGINEERING, False),
        ("software-b", Capability.SOFTWARE_ENGINEERING, True),
        ("visual-a", Capability.VISUAL_INTELLIGENCE, False),
        ("visual-b", Capability.VISUAL_INTELLIGENCE, True),
        ("quant-a", Capability.QUANTITATIVE_REASONING, False),
        ("social-a", Capability.SOCIAL_PRAGMATIC, True),
    ]
    benchmarks = [
        BenchmarkDefinition(
            id=benchmark_id,
            name=benchmark_id,
            family_id=f"family-{index}",
            published_at=date(2025, 1, 1),
            sealed_test=sealed,
            protocol_quality=0.95,
            reliability=0.95,
            contamination_resistance=0.95 if sealed else 0.70,
            capability_loadings={capability: 1.0},
        )
        for index, (benchmark_id, capability, sealed) in enumerate(benchmark_specs)
    ]

    observations: list[BenchmarkObservation] = []
    for model_index, model in enumerate(models):
        general = -1.8 + 0.4 * model_index
        specialist = {
            Capability.SOFTWARE_ENGINEERING: 1.2 if model.id == "m7" else 0.0,
            Capability.VISUAL_INTELLIGENCE: 1.0 if model.id == "m2" else 0.0,
        }
        for benchmark_index, (benchmark_id, capability, sealed) in enumerate(
            benchmark_specs
        ):
            score = 50.0 + 8.0 * general + 6.0 * specialist.get(capability, 0.0)
            harness = None
            if benchmark_id == "fluid-a":
                harness = "harness-a" if model_index % 2 == 0 else "harness-b"
                if harness == "harness-b":
                    score += 3.5
            source_url = (
                "https://eval-a.example.com/result"
                if benchmark_index % 2 == 0
                else "https://eval-b.example.com/result"
            )
            if model.id == "m6" and "eval-a" in source_url:
                score += 2.5
            if model.id == "m9" and sealed:
                score -= 5.0
            observations.append(
                BenchmarkObservation(
                    model_id=model.id,
                    benchmark_id=benchmark_id,
                    score=score,
                    evaluated_at=date(2026, 7, 1),
                    source_grade=SourceGrade.A,
                    harness=harness,
                    source_url=source_url,
                )
            )
    return models, benchmarks, observations


def test_hierarchical_estimator_recovers_general_and_domain_signal() -> None:
    models, benchmarks, observations = _synthetic_problem()
    result = fit_hierarchical(
        models,
        benchmarks,
        observations,
        config=HierarchicalConfig(
            minimum_models_per_benchmark=5,
            minimum_benchmarks_per_model=5,
            minimum_families_per_model=5,
            ridge_domain=1.5,
            ridge_ecosystem=3.0,
        ),
        as_of=date(2026, 9, 4),
    )

    by_model = {row.model_id: row for row in result.models}
    assert result.converged
    assert len(result.models) == 10
    assert result.weighted_r2 > result.general_only_r2
    assert by_model["m9"].general_z > by_model["m0"].general_z
    assert by_model["m7"].domain_residuals["software_engineering"] > 0
    assert by_model["m2"].domain_residuals["visual_intelligence"] > 0
    assert by_model["m9"].novelty_deviation < 0
    assert by_model["m9"].ci_low < by_model["m9"].general_z < by_model["m9"].ci_high


def test_hierarchical_estimator_separates_harness_and_ecosystem_effects() -> None:
    models, benchmarks, observations = _synthetic_problem()
    result = fit_hierarchical(
        models,
        benchmarks,
        observations,
        config=HierarchicalConfig(
            minimum_models_per_benchmark=5,
            minimum_benchmarks_per_model=5,
            minimum_families_per_model=5,
            ridge_harness=1.5,
            ridge_ecosystem=2.0,
        ),
        as_of=date(2026, 9, 4),
    )

    harness = {
        (row.benchmark_id, row.harness): row.effect_z for row in result.harness_effects
    }
    assert harness[("fluid-a", "harness-b")] > harness[("fluid-a", "harness-a")]

    ecosystem = {
        (row.model_id, row.ecosystem): row.effect_z for row in result.ecosystem_effects
    }
    assert ecosystem[("m6", "example.com")] == 0.0
