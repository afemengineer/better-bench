# ruff: noqa: I001
from __future__ import annotations

from datetime import date

from better_bench.io import load_observations


CUTOVERS = {
    "qwen3.8-max": (date(2026, 9, 2), "Qwen3.8-Max-0902"),
    "deepseek-v4-flash": (date(2026, 7, 31), "DeepSeek-V4-Flash-0731"),
    "deepseek-v4-pro": (date(2026, 8, 13), "DeepSeek-V4-Pro-0813"),
}

observations = load_observations(
    "data/current",
    include_unresolved_revisions=True,
)

print("model\tbenchmark\tevaluated_at\trevision\trevision_at\tstatus")
for row in observations:
    cutover = CUTOVERS.get(row.model_id)
    if cutover is None:
        continue
    cutover_at, expected_revision = cutover
    evaluated_at = row.evaluated_at
    revision_at = row.model_revision_at
    revision = row.model_revision

    if evaluated_at is None:
        status = "date_unknown"
    elif revision is not None or revision_at is not None:
        if revision == expected_revision and revision_at == cutover_at:
            status = "pinned_current"
        else:
            status = "pinned_other_revision"
    elif evaluated_at >= cutover_at:
        status = "post_cutover_unpinned"
    else:
        status = "pre_cutover_legacy_unpinned"

    print(
        f"{row.model_id}\t{row.benchmark_id}\t"
        f"{evaluated_at.isoformat() if evaluated_at else '-'}\t"
        f"{revision or '-'}\t{revision_at.isoformat() if revision_at else '-'}\t{status}"
    )
