# ruff: noqa: I001
from datetime import date
from pathlib import Path

from better_bench.estimator import EstimatorConfig, cross_validate_estimator, fit_estimator
from better_bench.general_ranking import build_general_intelligence_ranking
from better_bench.io import load_adoption, load_benchmarks, load_models, load_observations
from better_bench.provisional import project_provisional_models


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
official_ranking = build_general_intelligence_ranking(
    models,
    benchmarks,
    observations,
    result,
    adoption,
    as_of=date(2026, 9, 4),
)
provisional = project_provisional_models(
    models,
    benchmarks,
    observations,
    result,
    adoption,
    as_of=date(2026, 9, 4),
    minimum_families=config.minimum_families_per_model,
    ridge=config.ridge_general,
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
print(
    f"OFFICIAL_RANKING method={official_ranking.method} "
    f"benchmarks={official_ranking.benchmark_count} families={official_ranking.family_count} "
    f"weighted_agreement={100 * official_ranking.weighted_agreement:.2f}%"
)
print("\n[official]")
print("rank\tmodel\trank_band\tcalibration_z\tfamilies\teffective\tcoverage")
calibration_by_id = {row.model_id: row for row in result.models}
for ranked in official_ranking.models:
    row = calibration_by_id[ranked.model_id]
    print(
        f"{ranked.rank}\t{ranked.model_id}\t[{ranked.rank_low},{ranked.rank_high}]\t"
        f"{row.general_z:+.4f}\t{row.family_count}\t{row.effective_evidence:.3f}\t"
        f"{100 * row.coverage:.1f}%"
    )

print("\n[provisional]")
print("model\tprojected_z\tfamilies\teffective\tcoverage\tconfidence\treason")
for row in provisional:
    projected = "NA" if row.projected_general_z is None else f"{row.projected_general_z:+.4f}"
    print(
        f"{row.model_id}\t{projected}\t{row.family_count}\t{row.effective_evidence:.3f}\t"
        f"{100 * row.coverage:.1f}%\t{row.confidence}\t{row.reason}"
    )

print("\n[spark_lineage]")
official = {row.model_id: row for row in result.models}
provisional_by_id = {row.model_id: row for row in provisional}
rank_by_id = official_ranking.by_model
for model_id in ("muse-spark-1.1", "muse-spark-1.2", "muse-spark-1.3"):
    if model_id in official:
        row = official[model_id]
        ranked = rank_by_id[model_id]
        print(
            f"{model_id}\tofficial\trank={ranked.rank}\t[{ranked.rank_low},{ranked.rank_high}]\t"
            f"calibration_z={row.general_z:+.4f}\t{row.family_count}\t"
            f"{row.effective_evidence:.3f}\t{100 * row.coverage:.1f}%"
        )
    elif model_id in provisional_by_id:
        row = provisional_by_id[model_id]
        projected = "NA" if row.projected_general_z is None else f"{row.projected_general_z:+.4f}"
        print(
            f"{model_id}\tprovisional\t{projected}\t{row.family_count}\t"
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
    print("rank\tmodel\ttaxonomy_z\tcalibration_z\tresidual_pt\tfamilies\tcoverage")
    for rank, row in enumerate(ranked[:5], start=1):
        residual = row.domain_residual_points[capability]
        taxonomy_z = row.general_z + residual / 10.0
        print(
            f"{rank}\t{row.model_id}\t{taxonomy_z:+.4f}\t{row.general_z:+.4f}\t"
            f"{residual:+.3f}\t{row.family_count}\t{100 * row.coverage:.1f}%"
        )
