from collections import defaultdict
from datetime import date

from better_bench.benchmark_quality import BenchmarkTier, rank_benchmarks
from better_bench.io import load_adoption, load_benchmarks, load_models, load_observations


benchmarks = load_benchmarks("data/current")
models = load_models("data/current")
observations = load_observations("data/current")
adoption = load_adoption("data/current")

benchmark_by_id = {benchmark.id: benchmark for benchmark in benchmarks}
quality_rows = rank_benchmarks(
    benchmarks,
    models,
    observations,
    adoption,
    as_of=date(2026, 9, 4),
)
quality_by_id = {row.benchmark_id: row for row in quality_rows}

observed_by_model: dict[str, set[str]] = defaultdict(set)
for row in observations:
    observed_by_model[row.model_id].add(row.benchmark_id)

priority_tiers = {BenchmarkTier.CORE, BenchmarkTier.EMERGING}
priority_rows = [row for row in quality_rows if row.tier in priority_tiers]

print(
    "model\tbenchmarks\tfamilies\tcore_or_emerging\tpriority_missing"
)
for model in models:
    seen = observed_by_model.get(model.id, set())
    families = {
        benchmark_by_id[benchmark_id].family_id or benchmark_id
        for benchmark_id in seen
        if benchmark_id in benchmark_by_id
    }
    priority_seen = sum(
        1
        for benchmark_id in seen
        if benchmark_id in quality_by_id
        and quality_by_id[benchmark_id].tier in priority_tiers
    )
    missing = [
        row.benchmark_id
        for row in priority_rows
        if row.benchmark_id not in seen
    ][:6]
    print(
        f"{model.id}\t{len(seen)}\t{len(families)}\t{priority_seen}\t"
        + ",".join(missing)
    )
