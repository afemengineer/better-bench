# ruff: noqa: I001
from datetime import date
from pathlib import Path

from better_bench.estimator import EstimatorConfig, cross_validate_estimator, fit_estimator
from better_bench.io import load_adoption, load_benchmarks, load_models, load_observations


DATA = Path("data/current")
models = load_models(DATA)
benchmarks = load_benchmarks(DATA)
observations = load_observations(DATA)
adoption = load_adoption(DATA)
config = EstimatorConfig(jackknife_uncertainty=False)

result = fit_estimator(
    models,
    benchmarks,
    observations,
    adoption,
    config=config,
    as_of=date(2026, 9, 4),
)
cv = cross_validate_estimator(
    models,
    benchmarks,
    observations,
    adoption,
    config=config,
    folds=5,
    as_of=date(2026, 9, 4),
)

print(
    f"POINT_ESTIMATOR models={len(result.retained_models)} benchmarks={len(result.retained_benchmarks)} "
    f"observed={result.observed_cells}/{result.possible_cells} density={100 * result.density:.1f}% "
    f"fit_rmse={result.weighted_rmse_points:.3f} fit_r2={result.weighted_r2_vs_benchmark_mean:.4f}"
)
print(
    f"POINT_CV heldout={cv.heldout_observations} rmse={cv.model_rmse_points:.3f} "
    f"baseline={cv.benchmark_only_rmse_points:.3f} gain={100 * cv.relative_rmse_improvement:+.2f}% "
    f"r2={cv.residual_r2:+.4f} rho={cv.residual_spearman:+.4f}"
)
print("\n[overall]")
print("rank\tmodel\tgeneral_z\tfamilies\teffective\tcoverage")
for rank, row in enumerate(result.models, start=1):
    print(
        f"{rank}\t{row.model_id}\t{row.general_z:+.4f}\t{row.family_count}\t"
        f"{row.effective_evidence:.3f}\t{100 * row.coverage:.1f}%"
    )

capabilities = list(result.models[0].domain_residual_points)
for capability in capabilities:
    ranked = sorted(
        result.models,
        key=lambda row: row.general_z + row.domain_residual_points[capability] / 10.0,
        reverse=True,
    )
    print(f"\n[{capability}]")
    print("rank\tmodel\ttaxonomy_z\tgeneral_z\tresidual_pt\tfamilies\tcoverage")
    for rank, row in enumerate(ranked[:5], start=1):
        residual = row.domain_residual_points[capability]
        taxonomy_z = row.general_z + residual / 10.0
        print(
            f"{rank}\t{row.model_id}\t{taxonomy_z:+.4f}\t{row.general_z:+.4f}\t"
            f"{residual:+.3f}\t{row.family_count}\t{100 * row.coverage:.1f}%"
        )
