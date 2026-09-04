# ruff: noqa: I001
from datetime import date
from pathlib import Path

from better_bench.hierarchical import HierarchicalConfig
from better_bench.hierarchical_validation import cross_validate_hierarchical
from better_bench.io import load_adoption, load_benchmarks, load_models, load_observations


DATA = Path("data/current")
result = cross_validate_hierarchical(
    load_models(DATA),
    load_benchmarks(DATA),
    load_observations(DATA),
    load_adoption(DATA),
    config=HierarchicalConfig(
        minimum_models_per_benchmark=5,
        minimum_benchmarks_per_model=5,
        minimum_families_per_model=5,
    ),
    folds=5,
    as_of=date(2026, 9, 4),
)

print(
    f"model-family CV: folds={result.folds} heldout={result.heldout_observations} "
    f"full_rmse={result.full_rmse:.3f} g_only_rmse={result.general_only_rmse:.3f} "
    f"rmse_gain={100 * result.relative_rmse_improvement:+.1f}%"
)
print(
    f"heldout_r2: full={result.full_r2:+.3f} "
    f"g_only={result.general_only_r2:+.3f}"
)
print("largest held-out model errors")
print("model\theldout\tfull_rmse\tg_only_rmse\tbias")
for row in result.model_diagnostics[:15]:
    print(
        f"{row.model_id}\t{row.heldout_observations}\t{row.full_rmse:.3f}\t"
        f"{row.general_only_rmse:.3f}\t{row.full_bias:+.3f}"
    )
