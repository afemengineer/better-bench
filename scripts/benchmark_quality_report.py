from datetime import date
from pathlib import Path

from better_bench.io import (
    load_adoption,
    load_benchmarks,
    load_models,
    load_observations,
)
from better_bench.benchmark_quality import rank_benchmarks


DATA = Path("data/current")
rows = rank_benchmarks(
    load_benchmarks(DATA),
    load_models(DATA),
    load_observations(DATA),
    load_adoption(DATA),
    as_of=date(2026, 9, 4),
)

print(
    "tier\timportance\tadoption\tdiscrimination\tintegrity\tquality\t"
    "independence\tmodels\tbenchmark"
)
for row in rows:
    print(
        f"{row.tier.value}\t{row.importance:.3f}\t{row.adoption:.3f}\t"
        f"{row.discrimination:.3f}\t{row.integrity:.3f}\t{row.quality:.3f}\t"
        f"{row.independence:.3f}\t{row.effective_leaderboard_models}\t{row.benchmark_id}"
    )
