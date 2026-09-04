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


def load_benchmarks(path: str | Path) -> list[BenchmarkDefinition]:
    return _load_yaml_list(path, BenchmarkDefinition)


def load_models(path: str | Path) -> list[ModelDefinition]:
    return _load_yaml_list(path, ModelDefinition)


def load_observations(path: str | Path) -> list[BenchmarkObservation]:
    path = Path(path)
    if path.suffix.lower() in {".yaml", ".yml"}:
        return _load_yaml_list(path, BenchmarkObservation)
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        cleaned = []
        for row in rows:
            cleaned.append({key: value for key, value in row.items() if value not in (None, "")})
        return TypeAdapter(list[BenchmarkObservation]).validate_python(cleaned)
    raise ValueError(f"Unsupported observation format: {path.suffix}")
