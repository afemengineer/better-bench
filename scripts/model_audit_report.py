from __future__ import annotations

from collections import defaultdict
from datetime import date

from better_bench.estimator import EstimatorConfig, fit_estimator
from better_bench.indexing import build_better_bench_index
from better_bench.io import load_adoption, load_benchmarks, load_models, load_observations
from better_bench.provisional import project_provisional_models


AS_OF = date(2026, 9, 4)

models = load_models("data/current")
benchmarks = load_benchmarks("data/current")
observations = load_observations("data/current")
adoption = load_adoption("data/current")

estimator = fit_estimator(
    models,
    benchmarks,
    observations,
    adoption,
    config=EstimatorConfig(jackknife_uncertainty=False),
    as_of=AS_OF,
)
index = build_better_bench_index(
    models,
    benchmarks,
    observations,
    estimator,
    adoption,
    as_of=AS_OF,
)
provisional = project_provisional_models(
    models,
    benchmarks,
    observations,
    estimator,
    adoption,
    as_of=AS_OF,
)

benchmark_by_id = {row.id: row for row in benchmarks}
index_by_model = {row.model_id: (rank, row) for rank, row in enumerate(index, start=1)}
provisional_by_model = {row.model_id: row for row in provisional}

raw_benchmarks: dict[str, set[str]] = defaultdict(set)
raw_families: dict[str, set[str]] = defaultdict(set)
revision_dates: dict[str, set[date]] = defaultdict(set)
revision_labels: dict[str, set[str]] = defaultdict(set)
for row in observations:
    raw_benchmarks[row.model_id].add(row.benchmark_id)
    benchmark = benchmark_by_id.get(row.benchmark_id)
    if benchmark is not None:
        raw_families[row.model_id].add(benchmark.family_id or benchmark.id)
    if row.model_revision_at is not None:
        revision_dates[row.model_id].add(row.model_revision_at)
    if row.model_revision:
        revision_labels[row.model_id].add(row.model_revision)


def age_days(model) -> int | None:
    if model.released_at is None:
        return None
    return (AS_OF - model.released_at).days


def flags_for(model, rank, index_row, provisional_row) -> list[str]:
    flags: list[str] = []
    age = age_days(model)
    latest_revision = max(revision_dates.get(model.id, set()), default=None)
    if age is None:
        flags.append("release_unknown")
    elif age <= 7:
        flags.append("new_7d")
    elif age <= 30:
        flags.append("new_30d")
    elif age <= 60:
        flags.append("new_60d")
    if latest_revision is not None and (AS_OF - latest_revision).days <= 30:
        flags.append("recent_revision")
    if len(revision_dates.get(model.id, set())) > 1 or len(revision_labels.get(model.id, set())) > 1:
        flags.append("mixed_revisions")

    if index_row is not None:
        half_width = (index_row.ci_high - index_row.ci_low) / 2.0
        if index_row.family_count < 8:
            flags.append("sparse_retained")
        if index_row.effective_evidence < 7:
            flags.append("low_effective_evidence")
        if half_width > 4:
            flags.append("wide_interval")
        if rank is not None and rank <= 10:
            flags.append("top10")
    elif provisional_row is not None:
        flags.append("provisional")
        if provisional_row.family_count < 5:
            flags.append("below_rankability")
    else:
        flags.append("unscored")
    return flags


def priority_for(rank, index_row, provisional_row, flags: list[str]) -> str:
    flagged = set(flags)
    if rank is not None and rank <= 10 and flagged.intersection(
        {"new_7d", "new_30d", "sparse_retained", "wide_interval", "recent_revision"}
    ):
        return "URGENT"
    if provisional_row is not None and flagged.intersection(
        {"new_7d", "new_30d", "new_60d", "recent_revision"}
    ):
        return "URGENT"
    if rank is not None and rank <= 15 and flagged.intersection(
        {"new_60d", "sparse_retained", "wide_interval", "recent_revision"}
    ):
        return "HIGH"
    if index_row is not None and flagged.intersection(
        {"sparse_retained", "wide_interval", "mixed_revisions"}
    ):
        return "HIGH"
    if provisional_row is not None:
        return "HIGH" if "below_rankability" in flagged else "MEDIUM"
    return "MEDIUM" if flagged.intersection({"new_60d", "recent_revision"}) else "LOW"


rows = []
for model in models:
    official = index_by_model.get(model.id)
    rank = official[0] if official is not None else None
    index_row = official[1] if official is not None else None
    provisional_row = provisional_by_model.get(model.id)
    flags = flags_for(model, rank, index_row, provisional_row)
    priority = priority_for(rank, index_row, provisional_row, flags)
    latest_revision = max(revision_dates.get(model.id, set()), default=None)
    rows.append(
        (
            {"URGENT": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}[priority],
            rank if rank is not None else 999,
            model.id,
            priority,
            age_days(model),
            latest_revision,
            index_row,
            provisional_row,
            flags,
        )
    )

rows.sort(key=lambda item: (item[0], item[1], item[2]))

print(
    "priority\trank\tmodel\tBBI/proj_z\tage_d\traw_bench\traw_fam\tret_fam\t"
    "effective\t95%CI\tlatest_revision\tflags"
)
for _, rank_value, model_id, priority, age, latest_revision, index_row, provisional_row, flags in rows:
    rank_text = str(rank_value) if rank_value != 999 else "-"
    if index_row is not None:
        score = f"{index_row.index:.1f}"
        retained = str(index_row.family_count)
        effective = f"{index_row.effective_evidence:.2f}"
        interval = f"[{index_row.ci_low:.1f},{index_row.ci_high:.1f}]"
    elif provisional_row is not None and provisional_row.projected_general_z is not None:
        score = f"z={provisional_row.projected_general_z:+.3f}"
        retained = str(provisional_row.family_count)
        effective = f"{provisional_row.effective_evidence:.2f}"
        interval = "-"
    else:
        score = "-"
        retained = "0"
        effective = "0.00"
        interval = "-"
    print(
        f"{priority}\t{rank_text}\t{model_id}\t{score}\t"
        f"{age if age is not None else '-'}\t{len(raw_benchmarks[model_id])}\t"
        f"{len(raw_families[model_id])}\t{retained}\t{effective}\t{interval}\t"
        f"{latest_revision.isoformat() if latest_revision is not None else '-'}\t"
        f"{','.join(flags) if flags else '-'}"
    )
