# ruff: noqa: I001
from better_bench.io import load_benchmarks, load_models, load_observations


benchmarks = load_benchmarks("data/current")
models = load_models("data/current")
raw_observations = load_observations(
    "data/current",
    include_unresolved_revisions=True,
)
observations = load_observations("data/current")
families = {benchmark.family_id or benchmark.id for benchmark in benchmarks}
raw_pairs = {(row.model_id, row.benchmark_id) for row in raw_observations}
scoring_pairs = {(row.model_id, row.benchmark_id) for row in observations}

print(f"models={len(models)}")
print(f"benchmarks={len(benchmarks)}")
print(f"families={len(families)}")
print(f"observations={len(raw_observations)}")
print(f"scoring_observations={len(observations)}")
print(f"revision_excluded_observations={len(raw_observations) - len(observations)}")
print(f"unique_model_benchmark_pairs={len(raw_pairs)}")
print(f"scoring_unique_model_benchmark_pairs={len(scoring_pairs)}")
print(
    "registry_density="
    f"{100 * len(raw_pairs) / (len(models) * len(benchmarks)):.1f}%"
)
print(
    "scoring_registry_density="
    f"{100 * len(scoring_pairs) / (len(models) * len(benchmarks)):.1f}%"
)
