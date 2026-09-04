# ruff: noqa: I001
from better_bench.io import load_benchmarks, load_models, load_observations


benchmarks = load_benchmarks("data/current")
models = load_models("data/current")
observations = load_observations("data/current")
families = {benchmark.family_id or benchmark.id for benchmark in benchmarks}
observed_pairs = {(row.model_id, row.benchmark_id) for row in observations}

print(f"models={len(models)}")
print(f"benchmarks={len(benchmarks)}")
print(f"families={len(families)}")
print(f"observations={len(observations)}")
print(f"unique_model_benchmark_pairs={len(observed_pairs)}")
print(
    "registry_density="
    f"{100 * len(observed_pairs) / (len(models) * len(benchmarks)):.1f}%"
)
