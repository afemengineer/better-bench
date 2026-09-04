from __future__ import annotations

from datetime import date
from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, Field, field_validator, model_validator


class Capability(StrEnum):
    FLUID_REASONING = "fluid_reasoning"
    QUANTITATIVE_REASONING = "quantitative_reasoning"
    SCIENTIFIC_REASONING = "scientific_reasoning"
    KNOWLEDGE = "knowledge"
    LANGUAGE = "language"
    SOFTWARE_ENGINEERING = "software_engineering"
    TERMINAL_AGENCY = "terminal_agency"
    WEB_AGENCY = "web_agency"
    GUI_COMPUTER_USE = "gui_computer_use"
    VISUAL_INTELLIGENCE = "visual_intelligence"
    SPATIAL_INTELLIGENCE = "spatial_intelligence"
    LONG_CONTEXT = "long_context"
    PLANNING_AGENCY = "planning_agency"
    SOCIAL_PRAGMATIC = "social_pragmatic"


class SourceGrade(StrEnum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"


SOURCE_GRADE_WEIGHT: dict[SourceGrade, float] = {
    SourceGrade.A: 1.00,
    SourceGrade.B: 0.92,
    SourceGrade.C: 0.78,
    SourceGrade.D: 0.62,
    SourceGrade.E: 0.40,
}


class ModelDefinition(BaseModel):
    id: str
    name: str
    organization: str | None = None
    released_at: date | None = None
    training_cutoff: date | None = None
    knowledge_cutoff: str | None = None


class BenchmarkDefinition(BaseModel):
    id: str
    name: str
    version: str | None = None
    published_at: date
    public_since: date | None = None
    rotating: bool = False
    sealed_test: bool = False
    higher_is_better: bool = True
    score_floor: float = 0.0
    score_ceiling: float = 100.0
    protocol_quality: Annotated[float, Field(ge=0.0, le=1.0)] = 0.8
    reliability: Annotated[float, Field(ge=0.0, le=1.0)] = 0.8
    capability_loadings: dict[Capability, float]
    notes: str | None = None

    @field_validator("version", mode="before")
    @classmethod
    def normalize_yaml_date_version(cls, value: object) -> object:
        """Preserve version identifiers when YAML parses an ISO-looking string as a date."""
        if isinstance(value, date):
            return value.isoformat()
        return value

    @model_validator(mode="after")
    def validate_definition(self) -> BenchmarkDefinition:
        if self.score_ceiling <= self.score_floor:
            raise ValueError("score_ceiling must be greater than score_floor")
        if not self.capability_loadings:
            raise ValueError("capability_loadings cannot be empty")
        if any(weight < 0 for weight in self.capability_loadings.values()):
            raise ValueError("capability loadings must be non-negative")
        total = sum(self.capability_loadings.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"capability loadings must sum to 1.0, got {total:.6f}")
        return self


class BenchmarkObservation(BaseModel):
    model_id: str
    benchmark_id: str
    score: float
    evaluated_at: date | None = None
    source_grade: SourceGrade = SourceGrade.D
    harness: str | None = None
    reasoning_effort: str | None = None
    token_budget: int | None = None
    source_url: str | None = None
    notes: str | None = None
    model_revision: str | None = None
    model_revision_at: date | None = None


class CapabilityScore(BaseModel):
    capability: Capability
    score: float | None
    ci_low: float | None
    ci_high: float | None
    evidence_weight: float
    benchmark_count: int


class ModelScore(BaseModel):
    model_id: str
    model_name: str
    general_score: float | None
    ci_low: float | None
    ci_high: float | None
    coverage: float
    effective_benchmarks: float
    capability_scores: list[CapabilityScore]
