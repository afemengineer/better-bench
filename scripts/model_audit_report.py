# ruff: noqa: I001
from __future__ import annotations

from collections import defaultdict
from datetime import date
from urllib.parse import urlparse

from better_bench.estimator import EstimatorConfig, fit_estimator
from better_bench.hierarchical import HierarchicalConfig
from better_bench.hierarchical_validation import cross_validate_hierarchical
from better_bench.indexing import build_better_bench_index
from better_bench.io import load_adoption, load_benchmarks, load_models, load_observations
from better_bench.provisional import project_provisional_models


AS_OF = date(2026, 9, 4)
KNOWN_CUTOVERS = {
    "qwen3.8-max": date(2026, 9, 2),
    "deepseek-v4-flash": date(2026, 7, 31),
    "deepseek-v4-pro": date(2026, 8, 13),
}

models = load_models("data/current")
benchmarks = load_benchmarks("data/current")
raw_observations = load_observations(
    "data/current",
    include_unresolved_revisions=True,
)
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
cv = cross_validate_hierarchical(
    models,
    benchmarks,
    observations,
    adoption,
    config=HierarchicalConfig(
        minimum_models_per_benchmark=5,
        minimum_benchmarks_per_model=5,
        minimum_families_per_model=5,
    ),
    folds=5,
    as_of=AS_OF,
)

benchmark_by_id = {row.id: row for row in benchmarks}
index_by_model = {row.model_id: (rank, row) for rank, row in enumerate(index, start=1)}
provisional_by_model = {row.model_id: row for row in provisional}
cv_by_model = {row.model_id: row for row in cv.model_diagnostics}

raw_benchmarks: dict[str, set[str]] = defaultdict(set)
raw_families: dict[str, set[str]] = defaultdict(set)
source_domains: dict[str, set[str]] = defaultdict(set)
harnesses: dict[str, set[str]] = defaultdict(set)
revision_dates: dict[str, set[date]] = defaultdict(set)
revision_labels: dict[str, set[str]] = defaultdict(set)
unpinned_post_cutover: dict[str, int] = defaultdict(int)
legacy_pre_cutover: dict[str, int] = defaultdict(int)


def root_domain(url: str | None) -> str | None:
    if not url:
        return None
    host = (urlparse(url).hostname or "").lower().removeprefix("www.")
    if not host:
        return None
    parts = host.split(".")
    return host if len(parts) <= 2 else ".".join(parts[-2:])


for row in raw_observations:
    raw_benchmarks[row.model_id].add(row.benchmark_id)
    benchmark = benchmark_by_id.get(row.benchmark_id)
    if benchmark is not None:
        raw_families[row.model_id].add(benchmark.family_id or benchmark.id)
    domain = root_domain(row.source_url)
    if domain:
        source_domains[row.model_id].add(domain)
    if row.harness:
        harnesses[row.model_id].add(row.harness)
    if row.model_revision_at is not None:
        revision_dates[row.model_id].add(row.model_revision_at)
    if row.model_revision:
        revision_labels[row.model_id].add(row.model_revision)

    cutover = KNOWN_CUTOVERS.get(row.model_id)
    if cutover is not None and row.evaluated_at is not None:
        pinned = row.model_revision is not None or row.model_revision_at is not None
        if row.evaluated_at >= cutover and not pinned:
            unpinned_post_cutover[row.model_id] += 1
        elif row.evaluated_at < cutover and not pinned:
            legacy_pre_cutover[row.model_id] += 1


def age_days(model) -> int | None:
    if model.released_at is None:
        return None
    return (AS_OF - model.released_at).days


def flags_for(model, rank, index_row, provisional_row) -> list[str]:
    flags: list[str] = []
    age = age_days(model)
    latest_revision = max(revision_dates.get(model.id, set()), default=None)
    cv_row = cv_by_model.get(model.id)

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
    if unpinned_post_cutover.get(model.id, 0):
        flags.append("revision_unresolved")
    if legacy_pre_cutover.get(model.id, 0) and unpinned_post_cutover.get(model.id, 0):
        flags.append("cross_revision_mix")

    if len(source_domains[model.id]) < 3 and len(raw_families[model.id]) >= 5:
        flags.append("source_concentrated")
    if cv_row is not None:
        if cv_row.full_rmse >= 3.0:
            flags.append("high_profile_misfit")
        elif cv_row.full_rmse >= 2.0:
            flags.append("moderate_profile_misfit")
        if abs(cv_row.full_bias) >= 1.0:
            flags.append("systematic_cv_bias")

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
    severe = {
        "wide_interval",
        "high_profile_misfit",
        "revision_unresolved",
        "cross_revision_mix",
    }
    thin = {"sparse_retained", "low_effective_evidence", "source_concentrated"}
    recent = {"new_7d", "new_30d", "recent_revision"}

    if index_row is not None and rank is not None and rank <= 15:
        if flagged.intersection(severe):
            return "URGENT"
        if flagged.intersection(recent) and flagged.intersection(thin):
            return "URGENT"
        if flagged.intersection(recent):
            return "HIGH"
        if flagged.intersection(thin):
            return "HIGH"
    if provisional_row is not None and flagged.intersection(recent | severe):
        return "URGENT"
    if index_row is not None and flagged.intersection(severe):
        return "HIGH"
    if index_row is not None and flagged.intersection(thin):
        return "HIGH"
    if provisional_row is not None:
        return "HIGH" if "below_rankability" in flagged else "MEDIUM"
    if flagged.intersection({"new_60d", "moderate_profile_misfit", "systematic_cv_bias"}):
        return "MEDIUM"
    return "LOW"


rows = []
for model in models:
    official = index_by_model.get(model.id)
    rank = official[0] if official is not None else None
    index_row = official[1] if official is not None else None
    provisional_row = provisional_by_model.get(model.id)
    flags = flags_for(model, rank, index_row, provisional_row)
    priority = priority_for(rank, index_row, provisional_row, flags)
    latest_revision = max(revision_dates.get(model.id, set()), default=None)
    cv_row = cv_by_model.get(model.id)
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
            cv_row,
            flags,
        )
    )

rows.sort(key=lambda item: (item[0], item[1], item[2]))

print(
    "priority\trank\tmodel\tBBI/proj_z\tage_d\traw_bench\traw_fam\tret_fam\t"
    "effective\tsources\tharnesses\tCV_RMSE\tCV_bias\t95%CI\trevision_at\tflags"
)
for (
    _, rank_value, model_id, priority, age, latest_revision, index_row,
    provisional_row, cv_row, flags,
) in rows:
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
    cv_rmse = f"{cv_row.full_rmse:.2f}" if cv_row is not None else "-"
    cv_bias = f"{cv_row.full_bias:+.2f}" if cv_row is not None else "-"
    print(
        f"{priority}\t{rank_text}\t{model_id}\t{score}\t"
        f"{age if age is not None else '-'}\t{len(raw_benchmarks[model_id])}\t"
        f"{len(raw_families[model_id])}\t{retained}\t{effective}\t"
        f"{len(source_domains[model_id])}\t{len(harnesses[model_id])}\t"
        f"{cv_rmse}\t{cv_bias}\t{interval}\t"
        f"{latest_revision.isoformat() if latest_revision is not None else '-'}\t"
        f"{','.join(flags) if flags else '-'}"
    )
