# ruff: noqa: I001
from datetime import date
from pathlib import Path

from better_bench.hierarchical import HierarchicalConfig, fit_hierarchical
from better_bench.io import load_adoption, load_benchmarks, load_models, load_observations


DATA = Path("data/current")
result = fit_hierarchical(
    load_models(DATA),
    load_benchmarks(DATA),
    load_observations(DATA),
    load_adoption(DATA),
    config=HierarchicalConfig(
        minimum_models_per_benchmark=5,
        minimum_benchmarks_per_model=5,
        minimum_families_per_model=5,
    ),
    as_of=date(2026, 9, 4),
)

print(
    f"hierarchical retained matrix: {len(result.retained_models)} models x "
    f"{len(result.retained_benchmarks)} benchmarks | {result.observed_cells}/"
    f"{result.possible_cells} observed ({100 * result.density:.1f}%)"
)
print(
    f"converged={result.converged} iterations={result.iterations} "
    f"weighted_rmse={result.weighted_rmse:.3f} weighted_r2={result.weighted_r2:.3f} "
    f"general_only_r2={result.general_only_r2:.3f} "
    f"global_novelty={result.global_novelty_effect:+.3f}"
)
print("model\tgeneral_z\tci_low\tci_high\tfamilies\teffective\tnovelty_dev\ttop_domain")
for row in result.models:
    top_domain, top_value = max(
        row.domain_residuals.items(), key=lambda item: abs(item[1])
    )
    print(
        f"{row.model_id}\t{row.general_z:+.3f}\t{row.ci_low:+.3f}\t"
        f"{row.ci_high:+.3f}\t{row.family_count}\t{row.effective_evidence:.2f}\t"
        f"{row.novelty_deviation:+.3f}\t{top_domain}:{top_value:+.3f}"
    )

print("top benchmark general loadings")
for row in result.benchmarks[:12]:
    print(
        f"{row.benchmark_id}\t{row.general_loading:+.3f}\t"
        f"weight={row.evidence_weight:.3f}\t{row.tier}"
    )

print("largest harness adjustments")
for row in result.harness_effects[:10]:
    print(
        f"{row.benchmark_id}\t{row.harness}\t{row.effect_z:+.3f}\tn={row.observations}"
    )

print("largest model x source-ecosystem adjustments")
for row in result.ecosystem_effects[:10]:
    print(
        f"{row.model_id}\t{row.ecosystem}\t{row.effect_z:+.3f}\tn={row.observations}"
    )
