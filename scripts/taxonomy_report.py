# ruff: noqa: I001
from datetime import date
from pathlib import Path

from better_bench.estimator import EstimatorConfig, fit_estimator
from better_bench.io import load_adoption, load_benchmarks, load_models, load_observations


DATA = Path("data/current")
models = load_models(DATA)
benchmarks = load_benchmarks(DATA)
observations = load_observations(DATA)
adoption = load_adoption(DATA)

result = fit_estimator(
    models,
    benchmarks,
    observations,
    adoption,
    config=EstimatorConfig(),
    as_of=date(2026, 9, 4),
)

print("taxonomy ranking uses exploratory score = general_z + descriptive_domain_residual_points / 10")
print("domain residuals do not feed the validated general ranking")

capabilities = list(result.models[0].domain_residual_points)
for capability in capabilities:
    ranked = sorted(
        result.models,
        key=lambda row: row.general_z + row.domain_residual_points[capability] / 10.0,
        reverse=True,
    )
    print(f"\n[{capability}]")
    print("rank\tmodel\ttaxonomy_z\tgeneral_z\tdomain_residual_pt\tfamilies\tcoverage")
    for rank, row in enumerate(ranked[:5], start=1):
        residual = row.domain_residual_points[capability]
        taxonomy_z = row.general_z + residual / 10.0
        print(
            f"{rank}\t{row.model_id}\t{taxonomy_z:+.3f}\t{row.general_z:+.3f}\t"
            f"{residual:+.2f}\t{row.family_count}\t{100 * row.coverage:.1f}%"
        )
