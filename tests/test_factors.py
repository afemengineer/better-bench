from __future__ import annotations

from datetime import date

from better_bench.factors import fit_missing_pca
from better_bench.schema import BenchmarkDefinition, BenchmarkObservation, Capability


def test_missing_pca_recovers_dominant_shared_factor() -> None:
    benchmarks = [
        BenchmarkDefinition(
            id=f"bench-{index}",
            name=f"Bench {index}",
            published_at=date(2026, 1, 1),
            capability_loadings={Capability.FLUID_REASONING: 1.0},
        )
        for index in range(4)
    ]
    abilities = {
        "m0": -2.0,
        "m1": -1.5,
        "m2": -1.0,
        "m3": -0.5,
        "m4": 0.5,
        "m5": 1.0,
        "m6": 1.5,
        "m7": 2.0,
    }
    slopes = [9.0, 12.0, 7.0, 10.0]
    observations: list[BenchmarkObservation] = []
    for model_index, (model_id, ability) in enumerate(abilities.items()):
        for benchmark_index, slope in enumerate(slopes):
            if (model_index + benchmark_index) % 7 == 0:
                continue
            observations.append(
                BenchmarkObservation(
                    model_id=model_id,
                    benchmark_id=f"bench-{benchmark_index}",
                    score=50.0 + slope * ability,
                )
            )

    result = fit_missing_pca(
        benchmarks,
        observations,
        rank=2,
        minimum_models_per_benchmark=5,
        minimum_benchmarks_per_model=2,
    )

    assert result.density < 1.0
    assert result.explained_variance[0] > 0.90
    factor_one = result.model_scores["factor_1"]
    assert factor_one["m7"] > factor_one["m4"] > factor_one["m1"]
