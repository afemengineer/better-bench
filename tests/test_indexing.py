from datetime import date

from better_bench.estimator import EstimatorConfig, fit_estimator
from better_bench.indexing import build_better_bench_index
from better_bench.schema import (
    BenchmarkDefinition,
    BenchmarkObservation,
    Capability,
    ModelDefinition,
    SourceGrade,
)


def test_index_is_monotonic_and_reports_family_sensitivity_variance() -> None:
    models = [
        ModelDefinition(id=f"m{i}", name=f"M{i}", organization=f"o{i}")
        for i in range(8)
    ]
    benchmarks = [
        BenchmarkDefinition(
            id=f"b{j}",
            name=f"B{j}",
            family_id=f"f{j}",
            published_at=date(2026, 1, 1),
            protocol_quality=0.95,
            reliability=0.95,
            contamination_resistance=0.9,
            capability_loadings={Capability.FLUID_REASONING: 1.0},
        )
        for j in range(6)
    ]
    observations = []
    for i, model in enumerate(models):
        ability = -1.4 + 0.4 * i
        for j, benchmark in enumerate(benchmarks):
            perturbation = 7.0 if model.id == "m6" and j == 0 else 0.0
            observations.append(
                BenchmarkObservation(
                    model_id=model.id,
                    benchmark_id=benchmark.id,
                    score=50.0 + 8.0 * ability + perturbation,
                    source_grade=SourceGrade.A,
                )
            )

    estimator = fit_estimator(
        models,
        benchmarks,
        observations,
        config=EstimatorConfig(jackknife_uncertainty=False),
        as_of=date(2026, 9, 4),
    )
    index = build_better_bench_index(
        models,
        benchmarks,
        observations,
        estimator,
        as_of=date(2026, 9, 4),
    )

    by_model = {row.model_id: row for row in index}
    assert by_model["m7"].index > by_model["m4"].index > by_model["m0"].index
    assert by_model["m7"].index == round(100.0 + 10.0 * by_model["m7"].general_z, 1)
    assert by_model["m6"].family_sensitivity_se_z > 0
    assert by_model["m6"].index_variance >= 0
    assert by_model["m6"].ci_low < by_model["m6"].index < by_model["m6"].ci_high
