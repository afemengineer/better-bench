from __future__ import annotations

from pathlib import Path

from better_bench.io import load_benchmarks, load_models, load_observations


def test_modular_dataset_directory_loads_root_and_snapshots(tmp_path: Path) -> None:
    (tmp_path / "benchmarks").mkdir()
    (tmp_path / "observations").mkdir()

    (tmp_path / "benchmarks.yaml").write_text(
        """
- id: base-bench
  name: Base Bench
  published_at: 2026-01-01
  capability_loadings:
    fluid_reasoning: 1.0
""".strip(),
        encoding="utf-8",
    )
    (tmp_path / "benchmarks" / "extra.yaml").write_text(
        """
- id: extra-bench
  name: Extra Bench
  published_at: 2026-02-01
  capability_loadings:
    quantitative_reasoning: 1.0
""".strip(),
        encoding="utf-8",
    )
    (tmp_path / "models.yaml").write_text(
        """
- id: model-a
  name: Model A
""".strip(),
        encoding="utf-8",
    )
    header = (
        "model_id,benchmark_id,score,evaluated_at,source_grade,harness,"
        "reasoning_effort,token_budget,source_url,notes\n"
    )
    (tmp_path / "observations.csv").write_text(
        header + "model-a,base-bench,70,2026-03-01,A,,,,,root\n",
        encoding="utf-8",
    )
    (tmp_path / "observations" / "extra.csv").write_text(
        header + "model-a,extra-bench,80,2026-03-01,B,,,,,snapshot\n",
        encoding="utf-8",
    )

    benchmarks = load_benchmarks(tmp_path)
    models = load_models(tmp_path)
    observations = load_observations(tmp_path)

    assert {row.id for row in benchmarks} == {"base-bench", "extra-bench"}
    assert [row.id for row in models] == ["model-a"]
    assert {(row.benchmark_id, row.score) for row in observations} == {
        ("base-bench", 70.0),
        ("extra-bench", 80.0),
    }
