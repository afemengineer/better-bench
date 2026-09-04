from datetime import date

from better_bench.benchmark_quality import BenchmarkTier, rank_benchmarks
from better_bench.schema import (
    BenchmarkAdoptionSnapshot,
    BenchmarkDefinition,
    BenchmarkObservation,
    Capability,
    ModelDefinition,
)


def _benchmark(
    benchmark_id: str,
    *,
    published: date,
    protocol: float = 0.9,
    reliability: float = 0.9,
    resistance: float | None = None,
) -> BenchmarkDefinition:
    return BenchmarkDefinition(
        id=benchmark_id,
        name=benchmark_id,
        published_at=published,
        public_since=published,
        protocol_quality=protocol,
        reliability=reliability,
        contamination_resistance=resistance,
        capability_loadings={Capability.FLUID_REASONING: 1.0},
    )


def test_popular_discriminative_benchmark_can_be_core() -> None:
    benchmark = _benchmark(
        "core",
        published=date(2026, 6, 1),
        resistance=0.9,
    )
    models = [
        ModelDefinition(id=f"m{i}", name=f"m{i}", organization=f"org{i % 6}")
        for i in range(12)
    ]
    observations = [
        BenchmarkObservation(model_id=f"m{i}", benchmark_id="core", score=15 + 6 * i)
        for i in range(12)
    ]
    adoption = [
        BenchmarkAdoptionSnapshot(
            benchmark_id="core",
            as_of=date(2026, 9, 4),
            leaderboard_model_count=24,
            leaderboard_org_count=6,
        )
    ]
    row = rank_benchmarks(
        [benchmark], models, observations, adoption, as_of=date(2026, 9, 4)
    )[0]
    assert row.tier == BenchmarkTier.CORE


def test_saturated_popular_benchmark_does_not_become_core() -> None:
    benchmark = _benchmark("saturated", published=date(2024, 1, 1))
    models = [ModelDefinition(id=f"m{i}", name=f"m{i}") for i in range(12)]
    observations = [
        BenchmarkObservation(
            model_id=f"m{i}", benchmark_id="saturated", score=97.0 + 0.1 * i
        )
        for i in range(12)
    ]
    adoption = [
        BenchmarkAdoptionSnapshot(
            benchmark_id="saturated",
            as_of=date(2026, 9, 4),
            leaderboard_model_count=100,
            leaderboard_org_count=8,
        )
    ]
    row = rank_benchmarks(
        [benchmark], models, observations, adoption, as_of=date(2026, 9, 4)
    )[0]
    assert row.tier == BenchmarkTier.DIAGNOSTIC


def test_small_fresh_high_quality_benchmark_is_emerging_not_core() -> None:
    benchmark = _benchmark(
        "new",
        published=date(2026, 8, 20),
        protocol=0.96,
        reliability=0.92,
        resistance=0.95,
    )
    models = [
        ModelDefinition(id=f"m{i}", name=f"m{i}", organization=f"org{i}")
        for i in range(6)
    ]
    observations = [
        BenchmarkObservation(model_id=f"m{i}", benchmark_id="new", score=20 + 12 * i)
        for i in range(6)
    ]
    row = rank_benchmarks(
        [benchmark], models, observations, as_of=date(2026, 9, 4)
    )[0]
    assert row.tier == BenchmarkTier.EMERGING
