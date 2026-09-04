from datetime import date

from better_bench.estimator import BenchmarkCalibration, EstimatorResult
from better_bench.provisional import project_provisional_models
from better_bench.schema import (
    BenchmarkDefinition,
    BenchmarkObservation,
    Capability,
    ModelDefinition,
    SourceGrade,
)


def test_sparse_model_is_projected_without_joining_official_ranking() -> None:
    models = [
        ModelDefinition(id="m0", name="M0"),
        ModelDefinition(id="m1", name="M1"),
        ModelDefinition(id="sparse", name="Sparse"),
    ]
    benchmarks = [
        BenchmarkDefinition(
            id="b1",
            name="B1",
            family_id="f1",
            published_at=date(2026, 1, 1),
            capability_loadings={Capability.FLUID_REASONING: 1.0},
        ),
        BenchmarkDefinition(
            id="b2",
            name="B2",
            family_id="f2",
            published_at=date(2026, 1, 1),
            capability_loadings={Capability.SOFTWARE_ENGINEERING: 1.0},
        ),
    ]
    observations = [
        BenchmarkObservation(model_id="m0", benchmark_id="b1", score=40, source_grade=SourceGrade.A),
        BenchmarkObservation(model_id="m1", benchmark_id="b1", score=60, source_grade=SourceGrade.A),
        BenchmarkObservation(model_id="sparse", benchmark_id="b1", score=70, source_grade=SourceGrade.A),
        BenchmarkObservation(model_id="m0", benchmark_id="b2", score=45, source_grade=SourceGrade.A),
        BenchmarkObservation(model_id="m1", benchmark_id="b2", score=55, source_grade=SourceGrade.A),
        BenchmarkObservation(model_id="sparse", benchmark_id="b2", score=75, source_grade=SourceGrade.A),
    ]
    estimator = EstimatorResult(
        models=[],
        benchmarks=[
            BenchmarkCalibration(
                benchmark_id="b1",
                family_id="f1",
                tier="supporting",
                evidence_weight=0.6,
                intercept_points=50.0,
                general_loading_points_per_z=10.0,
                observed_models=2,
            ),
            BenchmarkCalibration(
                benchmark_id="b2",
                family_id="f2",
                tier="supporting",
                evidence_weight=0.6,
                intercept_points=50.0,
                general_loading_points_per_z=10.0,
                observed_models=2,
            ),
        ],
        retained_models=["m0", "m1"],
        retained_benchmarks=["b1", "b2"],
        observed_cells=4,
        possible_cells=4,
        density=1.0,
        weighted_rmse_points=0.0,
        weighted_r2_vs_benchmark_mean=1.0,
        iterations=1,
        converged=True,
    )

    rows = project_provisional_models(
        models,
        benchmarks,
        observations,
        estimator,
        as_of=date(2026, 9, 4),
        minimum_families=5,
    )

    assert len(rows) == 1
    sparse = rows[0]
    assert sparse.model_id == "sparse"
    assert sparse.status == "provisional"
    assert sparse.family_count == 2
    assert sparse.projected_general_z is not None
    assert sparse.projected_general_z > 0
    assert "3 more independent benchmark families" in sparse.reason
