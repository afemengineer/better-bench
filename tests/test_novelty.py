from datetime import date

from better_bench.novelty import (
    ExposureTier,
    classify_exposure,
    comparable_benchmark_residuals,
)
from better_bench.schema import (
    BenchmarkDefinition,
    BenchmarkObservation,
    Capability,
    ModelDefinition,
)


def bench(id: str, pub: date, *, sealed: bool = False, rotating: bool = False) -> BenchmarkDefinition:
    return BenchmarkDefinition(
        id=id,
        name=id,
        published_at=pub,
        public_since=pub,
        sealed_test=sealed,
        rotating=rotating,
        capability_loadings={
            Capability.SOFTWARE_ENGINEERING: 0.7,
            Capability.TERMINAL_AGENCY: 0.3,
        },
    )


def test_exposure_strength_tiers() -> None:
    model = ModelDefinition(
        id="m",
        name="m",
        released_at=date(2026, 3, 1),
        training_cutoff=date(2026, 1, 15),
    )
    assert classify_exposure(model, bench("after-release", date(2026, 3, 2))).tier == ExposureTier.GUARANTEED_UNSEEN_POST_RELEASE
    assert classify_exposure(model, bench("after-cutoff", date(2026, 2, 1))).tier == ExposureTier.DISCLOSED_CUTOFF_UNSEEN
    assert classify_exposure(model, bench("sealed", date(2025, 1, 1), sealed=True)).tier == ExposureTier.SEALED_TEST
    no_cutoff = ModelDefinition(id="x", name="x", released_at=date(2026, 3, 1))
    assert classify_exposure(no_cutoff, bench("recent", date(2026, 2, 20)), likely_unseen_days=45).tier == ExposureTier.LIKELY_UNSEEN_SHORT_LEAD


def test_new_benchmark_underperformance_surfaces_as_negative_residual() -> None:
    models = [
        ModelDefinition(id="bad", name="bad", released_at=date(2026, 1, 15)),
        ModelDefinition(id="good", name="good", released_at=date(2026, 1, 15)),
        ModelDefinition(id="m3", name="m3", released_at=date(2026, 4, 1)),
        ModelDefinition(id="m4", name="m4", released_at=date(2026, 4, 1)),
        ModelDefinition(id="m5", name="m5", released_at=date(2026, 4, 1)),
        ModelDefinition(id="m6", name="m6", released_at=date(2026, 4, 1)),
        ModelDefinition(id="m7", name="m7", released_at=date(2026, 4, 1)),
    ]
    benches = [
        bench("old1", date(2025, 1, 1)),
        bench("old2", date(2025, 6, 1)),
        bench("new", date(2026, 2, 1)),
    ]
    base = {"bad": 85, "good": 82, "m3": 75, "m4": 68, "m5": 60, "m6": 52, "m7": 45}
    observations: list[BenchmarkObservation] = []
    for model_id, value in base.items():
        observations.append(BenchmarkObservation(model_id=model_id, benchmark_id="old1", score=value))
        observations.append(BenchmarkObservation(model_id=model_id, benchmark_id="old2", score=value - 2))
        new_score = 35 if model_id == "bad" else value - 4
        observations.append(BenchmarkObservation(model_id=model_id, benchmark_id="new", score=new_score))

    rows = comparable_benchmark_residuals(models, benches, observations, minimum_overlap=5)
    bad_new = next(row for row in rows if row.model_id == "bad" and row.benchmark_id == "new")
    assert bad_new.exposure_tier == ExposureTier.GUARANTEED_UNSEEN_POST_RELEASE
    assert bad_new.residual < -20
