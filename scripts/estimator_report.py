# ruff: noqa: I001
from datetime import date
from pathlib import Path

from better_bench.estimator import (
    EstimatorConfig,
    cross_validate_estimator,
    fit_estimator,
)
from better_bench.io import load_adoption, load_benchmarks, load_models, load_observations


DATA = Path("data/current")
models = load_models(DATA)
benchmarks = load_benchmarks(DATA)
observations = load_observations(DATA)
adoption = load_adoption(DATA)
config = EstimatorConfig()

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
    config=EstimatorConfig(jackknife_uncertainty=False),
    folds=5,
    as_of=date(2026, 9, 4),
)

print(
    f"validated estimator: {len(result.retained_models)} models x "
    f"{len(result.retained_benchmarks)} benchmarks | "
    f"{result.observed_cells}/{result.possible_cells} observed "
    f"({100 * result.density:.1f}%)"
)
print(
    f"fit: converged={result.converged} iterations={result.iterations} "
    f"rmse={result.weighted_rmse_points:.2f}pts "
    f"R2_vs_benchmark_mean={result.weighted_r2_vs_benchmark_mean:.3f}"
)
print(
    f"family-CV: heldout={cv.heldout_observations} "
    f"model_rmse={cv.model_rmse_points:.2f}pts "
    f"benchmark_only={cv.benchmark_only_rmse_points:.2f}pts "
    f"gain={100 * cv.relative_rmse_improvement:+.1f}% "
    f"residual_R2={cv.residual_r2:+.3f} residual_rho={cv.residual_spearman:+.3f}"
)
print(
    "model\tgeneral_z\tci_low\tci_high\tfamilies\teffective\t"
    "jackknife_se\ttop_descriptive_domain"
)
for row in result.models:
    domain, value = max(
        row.domain_residual_points.items(), key=lambda item: abs(item[1])
    )
    print(
        f"{row.model_id}\t{row.general_z:+.3f}\t{row.ci_low:+.3f}\t"
        f"{row.ci_high:+.3f}\t{row.family_count}\t{row.effective_evidence:.2f}\t"
        f"{row.family_jackknife_se:.3f}\t{domain}:{value:+.2f}pt"
    )

print("top learned benchmark discrimination")
for row in result.benchmarks[:12]:
    print(
        f"{row.benchmark_id}\t{row.general_loading_points_per_z:+.2f}pt/z\t"
        f"intercept={row.intercept_points:.1f}\tweight={row.evidence_weight:.3f}\t"
        f"{row.tier}"
    )
