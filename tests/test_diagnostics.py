from datetime import date

from better_bench.diagnostics import (
    benchmark_pair_residuals,
    pairwise_benchmark_diagnostics,
    taxonomy_fit,
)
from better_bench.schema import BenchmarkDefinition, BenchmarkObservation, Capability


def _benchmark(benchmark_id: str, capability: Capability) -> BenchmarkDefinition:
    return BenchmarkDefinition(
        id=benchmark_id,
        name=benchmark_id,
        published_at=date(2026, 1, 1),
        capability_loadings={capability: 1.0},
    )


def test_pairwise_diagnostics_detect_shared_signal() -> None:
    benchmarks = [
        _benchmark("a", Capability.SOFTWARE_ENGINEERING),
        _benchmark("b", Capability.SOFTWARE_ENGINEERING),
        _benchmark("c", Capability.SCIENTIFIC_REASONING),
    ]
    observations = []
    for i, (a, b, c) in enumerate(
        [(10, 12, 90), (20, 19, 20), (30, 31, 70), (40, 39, 40), (50, 52, 60)]
    ):
        model = f"m{i}"
        observations.extend(
            [
                BenchmarkObservation(model_id=model, benchmark_id="a", score=a),
                BenchmarkObservation(model_id=model, benchmark_id="b", score=b),
                BenchmarkObservation(model_id=model, benchmark_id="c", score=c),
            ]
        )

    rows = pairwise_benchmark_diagnostics(benchmarks, observations, minimum_overlap=4)
    ab = next(row for row in rows if {row.left, row.right} == {"a", "b"})
    assert ab.spearman > 0.9
    assert ab.loading_similarity == 1.0

    fit, count = taxonomy_fit(benchmarks, observations, minimum_overlap=4)
    assert count == 3
    assert fit is not None
    assert fit > 0.5


def test_residuals_surface_cross_benchmark_outlier() -> None:
    observations = []
    for i, (left, right) in enumerate([(10, 10), (20, 20), (30, 30), (40, 5), (50, 50)]):
        model = f"m{i}"
        observations.append(BenchmarkObservation(model_id=model, benchmark_id="left", score=left))
        observations.append(BenchmarkObservation(model_id=model, benchmark_id="right", score=right))

    rows = benchmark_pair_residuals(observations, "left", "right", minimum_overlap=4)
    assert rows
    assert rows[0].model_id == "m3"
    assert rows[0].residual < 0
