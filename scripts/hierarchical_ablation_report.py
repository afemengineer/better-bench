# ruff: noqa: I001
from datetime import date
from pathlib import Path

from better_bench.hierarchical import HierarchicalConfig
from better_bench.hierarchical_validation import cross_validate_hierarchical
from better_bench.io import load_adoption, load_benchmarks, load_models, load_observations


DATA = Path("data/current")
MODELS = load_models(DATA)
BENCHMARKS = load_benchmarks(DATA)
OBSERVATIONS = load_observations(DATA)
ADOPTION = load_adoption(DATA)
HUGE = 1e9

variants = {
    "domains": HierarchicalConfig(
        ridge_domain=3.0,
        ridge_harness=HUGE,
        ridge_ecosystem=HUGE,
        ridge_global_novelty=HUGE,
        ridge_model_novelty=HUGE,
    ),
    "harness": HierarchicalConfig(
        ridge_domain=HUGE,
        ridge_harness=4.0,
        ridge_ecosystem=HUGE,
        ridge_global_novelty=HUGE,
        ridge_model_novelty=HUGE,
    ),
    "ecosystem": HierarchicalConfig(
        ridge_domain=HUGE,
        ridge_harness=HUGE,
        ridge_ecosystem=6.0,
        ridge_global_novelty=HUGE,
        ridge_model_novelty=HUGE,
    ),
    "novelty": HierarchicalConfig(
        ridge_domain=HUGE,
        ridge_harness=HUGE,
        ridge_ecosystem=HUGE,
        ridge_global_novelty=2.0,
        ridge_model_novelty=5.0,
    ),
    "domains+harness": HierarchicalConfig(
        ridge_domain=3.0,
        ridge_harness=4.0,
        ridge_ecosystem=HUGE,
        ridge_global_novelty=HUGE,
        ridge_model_novelty=HUGE,
    ),
    "domains+novelty": HierarchicalConfig(
        ridge_domain=3.0,
        ridge_harness=HUGE,
        ridge_ecosystem=HUGE,
        ridge_global_novelty=2.0,
        ridge_model_novelty=5.0,
    ),
    "full": HierarchicalConfig(),
}

print("variant\tfull_rmse\tg_only_rmse\trmse_gain\tfull_r2\tg_only_r2")
for name, config in variants.items():
    result = cross_validate_hierarchical(
        MODELS,
        BENCHMARKS,
        OBSERVATIONS,
        ADOPTION,
        config=config,
        folds=5,
        as_of=date(2026, 9, 4),
    )
    print(
        f"{name}\t{result.full_rmse:.4f}\t{result.general_only_rmse:.4f}\t"
        f"{100 * result.relative_rmse_improvement:+.2f}%\t"
        f"{result.full_r2:+.4f}\t{result.general_only_r2:+.4f}"
    )
