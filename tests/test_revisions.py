from datetime import date

import pytest

from better_bench.revisions import is_scoring_eligible, scoring_observations
from better_bench.schema import BenchmarkObservation


def _row(
    model_id: str,
    evaluated_at: date | None,
    *,
    revision_at: date | None = None,
) -> BenchmarkObservation:
    return BenchmarkObservation(
        model_id=model_id,
        benchmark_id="benchmark",
        score=50.0,
        evaluated_at=evaluated_at,
        model_revision_at=revision_at,
    )


@pytest.mark.parametrize(
    ("model_id", "cutover"),
    [
        ("qwen3.8-max", date(2026, 9, 2)),
        ("deepseek-v4-flash", date(2026, 7, 31)),
        ("deepseek-v4-pro", date(2026, 8, 13)),
    ],
)
def test_unpinned_mutable_alias_is_only_safe_before_cutover(
    model_id: str,
    cutover: date,
) -> None:
    assert is_scoring_eligible(_row(model_id, cutover.replace(day=cutover.day - 1)))
    assert not is_scoring_eligible(_row(model_id, cutover))


def test_post_cutover_legacy_rerun_is_safe_when_revision_predates_cutover() -> None:
    row = _row(
        "qwen3.8-max",
        date(2026, 9, 4),
        revision_at=date(2026, 8, 3),
    )
    assert is_scoring_eligible(row)


def test_successor_snapshot_uses_its_own_identity() -> None:
    row = _row("qwen3.8-max-0902", date(2026, 9, 4))
    assert is_scoring_eligible(row)


def test_scoring_filter_preserves_safe_rows_and_drops_ambiguous_aliases() -> None:
    rows = [
        _row("qwen3.8-max", date(2026, 9, 4)),
        _row("qwen3.8-max-0902", date(2026, 9, 4)),
        _row("other-model", date(2026, 9, 4)),
    ]
    assert [row.model_id for row in scoring_observations(rows)] == [
        "qwen3.8-max-0902",
        "other-model",
    ]
