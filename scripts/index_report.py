# ruff: noqa: I001
from datetime import date

from better_bench.estimator import EstimatorConfig, cross_validate_estimator, fit_estimator
from better_bench.indexing import build_better_bench_index
from better_bench.io import load_adoption, load_benchmarks, load_models, load_observations


models = load_models("data/current")
benchmarks = load_benchmarks("data/current")
observations = load_observations("data/current")
adoption = load_adoption("data/current")
config = EstimatorConfig(jackknife_uncertainty=False)
estimator = fit_estimator(
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
index = build_better_bench_index(
    models,
    benchmarks,
    observations,
    estimator,
    adoption,
    as_of=date(2026, 9, 4),
)

print(
    f"BBI calibration=BBI-2026-09-05 models={len(index)} "
    f"ranking_method={index[0].ranking_method if index else '-'} "
    f"cv_rmse={cv.model_rmse_points:.3f} baseline={cv.benchmark_only_rmse_points:.3f} "
    f"gain={100 * cv.relative_rmse_improvement:+.2f}%"
)
print(
    "rank\tmodel\trank_band\tcalibration_BBI\t95%calibration_CI\tvariance\t"
    "families\tcoverage\tconfidence"
)
for row in index:
    print(
        f"{row.rank}\t{row.model_id}\t[{row.rank_low},{row.rank_high}]\t"
        f"{row.index:.1f}\t[{row.ci_low:.1f},{row.ci_high:.1f}]\t"
        f"{row.index_variance:.2f}\t{row.family_count}\t"
        f"{100 * row.coverage:.1f}%\t{row.confidence}"
    )
