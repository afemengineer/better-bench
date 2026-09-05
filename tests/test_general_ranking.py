from datetime import date
from types import SimpleNamespace

from better_bench.consensus_ranking import ConsensusRankingConfig, fit_consensus_ranking
from better_bench.ranking_evidence import prepare_ranking_evidence
from better_bench.schema import (
    BenchmarkDefinition,
    BenchmarkObservation,
    Capability,
    ModelDefinition,
    SourceGrade,
)


def _row(model: str, benchmark: str, family: str, score: float, weight: float = 1.0):
    return SimpleNamespace(
        model_id=model,
        benchmark_id=benchmark,
        family_id=family,
        score_points=float(score),
        weight=float(weight),
    )


def test_broad_ranking_panel_admits_two_model_benchmark_but_not_one_model_row() -> None:
    models = [
        ModelDefinition(id="a", name="A", organization="x"),
        ModelDefinition(id="b", name="B", organization="y"),
    ]
    benchmarks = [
        BenchmarkDefinition(
            id="shared",
            name="Shared",
            family_id="shared-family",
            published_at=date(2026, 1, 1),
            protocol_quality=0.95,
            reliability=0.95,
            contamination_resistance=0.9,
            capability_loadings={Capability.FLUID_REASONING: 1.0},
        ),
        BenchmarkDefinition(
            id="solo",
            name="Solo",
            family_id="solo-family",
            published_at=date(2026, 1, 1),
            protocol_quality=0.95,
            reliability=0.95,
            contamination_resistance=0.9,
            capability_loadings={Capability.FLUID_REASONING: 1.0},
        ),
    ]
    observations = [
        BenchmarkObservation(
            model_id="a", benchmark_id="shared", score=70, source_grade=SourceGrade.A
        ),
        BenchmarkObservation(
            model_id="b", benchmark_id="shared", score=60, source_grade=SourceGrade.A
        ),
        BenchmarkObservation(
            model_id="a", benchmark_id="solo", score=99, source_grade=SourceGrade.A
        ),
    ]

    panel = prepare_ranking_evidence(
        models,
        benchmarks,
        observations,
        [],
        rankable_model_ids=["a", "b"],
        as_of=date(2026, 9, 4),
    )

    assert panel.retained_benchmarks == ["shared"]
    assert {row.benchmark_id for row in panel.observations} == {"shared"}


def test_consensus_missing_hard_cell_cannot_make_sparse_model_beat_direct_winner() -> None:
    rows = [
        _row("strong", "b1", "f1", 70),
        _row("sparse", "b1", "f1", 60),
        _row("anchor", "b1", "f1", 45),
        _row("strong", "b2", "f2", 75),
        _row("sparse", "b2", "f2", 62),
        _row("anchor", "b2", "f2", 48),
        # The sparse model never ran this hard family. Absence creates no win.
        _row("strong", "hard", "f3", 32),
        _row("anchor", "hard", "f3", 18),
    ]

    result = fit_consensus_ranking(
        rows,
        ["strong", "sparse", "anchor"],
        None,
        score_unit_points=10.0,
        config=ConsensusRankingConfig(
            margin_temperature_points=10.0,
            minimum_shared_families=1,
            use_learned_discrimination=False,
        ),
    )

    assert result.rank["strong"] < result.rank["sparse"] < result.rank["anchor"]


def test_consensus_recovers_consistent_partial_order() -> None:
    rows = [
        _row("a", "x", "fx", 80),
        _row("b", "x", "fx", 70),
        _row("c", "x", "fx", 60),
        _row("a", "y", "fy", 78),
        _row("b", "y", "fy", 69),
        _row("c", "y", "fy", 55),
    ]
    result = fit_consensus_ranking(
        rows,
        ["a", "b", "c"],
        None,
        score_unit_points=10.0,
        config=ConsensusRankingConfig(use_learned_discrimination=False),
    )
    assert result.ranking == ["a", "b", "c"]
