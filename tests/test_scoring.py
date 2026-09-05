from datetime import date

import pytest

from better_bench.schema import (
    BenchmarkDefinition,
    BenchmarkObservation,
    Capability,
    ModelDefinition,
    SourceGrade,
)
from better_bench.scoring import contamination_weight, normalize_score, score_models


def benchmark(**overrides):
    values = {
        "id": "b1",
        "name": "Test",
        "published_at": date(2026, 1, 1),
        "score_floor": 0,
        "score_ceiling": 100,
        "capability_loadings": {Capability.SOFTWARE_ENGINEERING: 1.0},
    }
    values.update(overrides)
    return BenchmarkDefinition(**values)


def test_fixed_normalization_does_not_depend_on_frontier_model():
    assert normalize_score(75, benchmark()) == pytest.approx(75.0)


def test_public_benchmark_contamination_is_model_conditional():
    b = benchmark(public_since=date(2025, 1, 1))
    old_model = ModelDefinition(id="old", name="Old", training_cutoff=date(2024, 12, 1))
    new_model = ModelDefinition(id="new", name="New", training_cutoff=date(2026, 1, 1))
    as_of = date(2026, 9, 1)
    assert contamination_weight(b, old_model, as_of) == 1.0
    assert contamination_weight(b, new_model, as_of) < 1.0


def test_missing_domain_is_not_scored_as_zero_and_reduces_coverage():
    models = [ModelDefinition(id="m", name="Model")]
    benchmarks = [benchmark()]
    observations = [
        BenchmarkObservation(
            model_id="m", benchmark_id="b1", score=80, source_grade=SourceGrade.A
        )
    ]
    result = score_models(models, benchmarks, observations, as_of=date(2026, 9, 1))[0]
    assert result.general_score == pytest.approx(80.0)
    assert result.coverage == pytest.approx(1 / len(Capability), abs=1e-4)
    spatial = next(
        value for value in result.capability_scores if value.capability == Capability.SPATIAL_INTELLIGENCE
    )
    assert spatial.score is None
    assert result.ci_high - result.ci_low > 40


def test_benchmark_loadings_must_sum_to_one():
    with pytest.raises(ValueError):
        benchmark(capability_loadings={Capability.SOFTWARE_ENGINEERING: 0.5})
