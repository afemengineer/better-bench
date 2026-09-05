from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from .schema import BenchmarkObservation


@dataclass(frozen=True)
class MutableAliasPolicy:
    """Cutover after which a product alias no longer identifies one immutable model."""

    cutover_at: date
    successor_model_id: str


MUTABLE_ALIAS_POLICIES: dict[str, MutableAliasPolicy] = {
    "qwen3.8-max": MutableAliasPolicy(
        cutover_at=date(2026, 9, 2),
        successor_model_id="qwen3.8-max-0902",
    ),
    "deepseek-v4-flash": MutableAliasPolicy(
        cutover_at=date(2026, 7, 31),
        successor_model_id="deepseek-v4-flash-0731",
    ),
    "deepseek-v4-pro": MutableAliasPolicy(
        cutover_at=date(2026, 8, 13),
        successor_model_id="deepseek-v4-pro-0813",
    ),
}


def is_scoring_eligible(observation: BenchmarkObservation) -> bool:
    """Return whether an observation identifies one defensible statistical artifact.

    Rows on a known mutable product alias are safe before its cutover. After the
    cutover, an unpinned row could refer to either the legacy weights or the successor
    revision and is therefore excluded from scoring. A post-cutover measurement of the
    legacy artifact is still admissible when its immutable revision date is explicitly
    pinned to a date before the cutover.

    Successor snapshots use their own model ids and are unaffected by this gate.
    """

    policy = MUTABLE_ALIAS_POLICIES.get(observation.model_id)
    if policy is None:
        return True

    if observation.evaluated_at is not None and observation.evaluated_at < policy.cutover_at:
        return True

    if (
        observation.model_revision_at is not None
        and observation.model_revision_at < policy.cutover_at
    ):
        return True

    return False


def scoring_observations(
    observations: list[BenchmarkObservation],
) -> list[BenchmarkObservation]:
    """Filter raw evidence to rows safe for statistical estimation and ranking."""

    return [row for row in observations if is_scoring_eligible(row)]
