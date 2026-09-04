from __future__ import annotations

import csv
from pathlib import Path
from typing import TypeVar

import yaml
from pydantic import BaseModel, TypeAdapter

from .schema import BenchmarkDefinition, BenchmarkObservation, ModelDefinition

T = TypeVar("T", bound=BaseModel)


def _load_yaml_list(path: str | Path, model_type: type[T]) -> list[T]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or []
    if isinstance(raw, dict):
        raw = raw.get("items", raw.get("benchmarks", raw.get("models", raw)))
    if not isinstance(raw, list):
        raise ValueError(f"Expected a YAML list in {path}")
    return TypeAdapter(list[model_type]).validate_python(raw)


def _dataset_files(
    path: str | Path,
    *,
    family: str,
    suffixes: set[str],
) -> list[Path]:
    """Resolve either one data file or a modular dataset directory.

    A directory may contain the historical root file, e.g. ``benchmarks.yaml``,
    plus dated snapshots under ``benchmarks/``. This lets us append immutable
    provenance snapshots instead of rewriting one large source file.
    """
    resolved = Path(path)
    if resolved.is_file():
        return [resolved]
    if not resolved.is_dir():
        raise ValueError(f"Dataset path does not exist: {resolved}")

    files: list[Path] = []
    for suffix in sorted(suffixes):
        candidate = resolved / f"{family}{suffix}"
        if candidate.is_file():
            files.append(candidate)

    snapshot_dir = resolved / family
    if snapshot_dir.is_dir():
        files.extend(
            sorted(
                candidate
                for candidate in snapshot_dir.iterdir()
                if candidate.is_file() and candidate.suffix.lower() in suffixes
            )
        )

    if not files:
        allowed = ", ".join(sorted(suffixes))
        raise ValueError(
            f"No {family} files found under {resolved}; expected suffixes: {allowed}"
        )
    return files


def _ensure_unique(items: list[T], *, label: str) -> list[T]:
    seen: dict[str, Path | None] = {}
    for item in items:
        item_id = getattr(item, "id", None)
        if item_id is None:
            continue
        if item_id in seen:
            raise ValueError(f"Duplicate {label} id across dataset snapshots: {item_id}")
        seen[item_id] = None
    return items


def load_benchmarks(path: str | Path) -> list[BenchmarkDefinition]:
    files = _dataset_files(
        path,
        family="benchmarks",
        suffixes={".yaml", ".yml"},
    )
    rows: list[BenchmarkDefinition] = []
    for file in files:
        rows.extend(_load_yaml_list(file, BenchmarkDefinition))
    return _ensure_unique(rows, label="benchmark")


def load_models(path: str | Path) -> list[ModelDefinition]:
    files = _dataset_files(
        path,
        family="models",
        suffixes={".yaml", ".yml"},
    )
    rows: list[ModelDefinition] = []
    for file in files:
        rows.extend(_load_yaml_list(file, ModelDefinition))
    return _ensure_unique(rows, label="model")


def _load_observation_file(path: Path) -> list[BenchmarkObservation]:
    if path.suffix.lower() in {".yaml", ".yml"}:
        return _load_yaml_list(path, BenchmarkObservation)
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        cleaned = [
            {key: value for key, value in row.items() if value not in (None, "")}
            for row in rows
        ]
        return TypeAdapter(list[BenchmarkObservation]).validate_python(cleaned)
    raise ValueError(f"Unsupported observation format: {path.suffix}")


def load_observations(path: str | Path) -> list[BenchmarkObservation]:
    files = _dataset_files(
        path,
        family="observations",
        suffixes={".csv", ".yaml", ".yml"},
    )
    rows: list[BenchmarkObservation] = []
    for file in files:
        rows.extend(_load_observation_file(file))
    return rows
