# ruff: noqa: I001
from pathlib import Path

from better_bench.factors import fit_missing_pca
from better_bench.io import load_benchmarks, load_observations


DATA = Path("data/current")
result = fit_missing_pca(
    load_benchmarks(DATA),
    load_observations(DATA),
    rank=3,
    minimum_models_per_benchmark=5,
    minimum_benchmarks_per_model=5,
    minimum_families_per_model=5,
)

print(
    f"family-aware retained matrix: {len(result.models)} models x "
    f"{len(result.benchmarks)} benchmarks | {result.observed_cells}/"
    f"{result.possible_cells} observed ({100 * result.density:.1f}%)"
)
for index, explained in enumerate(result.explained_variance, start=1):
    print(f"rank {index} cumulative observed variance: {100 * explained:.1f}%")
print("factor_1_model_ranking")
for model_id, value in result.model_scores["factor_1"].sort_values(ascending=False).items():
    print(f"{model_id}\t{float(value):+.3f}")
