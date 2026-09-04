from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from enum import StrEnum

import numpy as np

from .schema import BenchmarkDefinition, BenchmarkObservation, Capability, ModelDefinition
from .scoring import normalize_score


class ExposureTier(StrEnum):
    GUARANTEED_UNSEEN_POST_RELEASE = "guaranteed_unseen_post_release"
    DISCLOSED_CUTOFF_UNSEEN = "disclosed_cutoff_unseen"
    SEALED_TEST = "sealed_test"
    ROTATING_LOW_EXPOSURE = "rotating_low_exposure"
    LIKELY_UNSEEN_SHORT_LEAD = "likely_unseen_short_lead"
    EXPOSURE_POSSIBLE = "exposure_possible"
    UNKNOWN = "unknown"


STRONG_UNSEEN = {
    ExposureTier.GUARANTEED_UNSEEN_POST_RELEASE,
    ExposureTier.DISCLOSED_CUTOFF_UNSEEN,
}
PROTECTED = {ExposureTier.SEALED_TEST, ExposureTier.ROTATING_LOW_EXPOSURE}
SUGGESTIVE_UNSEEN = {ExposureTier.LIKELY_UNSEEN_SHORT_LEAD}
POSSIBLY_EXPOSED = {ExposureTier.EXPOSURE_POSSIBLE}


@dataclass(frozen=True)
class ExposureAssessment:
    tier: ExposureTier
    rationale: str


@dataclass(frozen=True)
class NoveltyResidual:
    model_id: str
    benchmark_id: str
    observed_score: float
    predicted_score: float
    residual: float
    exposure_tier: ExposureTier
    exposure_group: str
    comparator_count: int
    comparator_weight: float


@dataclass(frozen=True)
class ModelNoveltySummary:
    model_id: str
    strong_unseen_count: int
    protected_count: int
    suggestive_unseen_count: int
    possible_exposure_count: int
    unknown_count: int
    broad_novel_mean_residual: float | None
    possible_exposure_mean_residual: float | None
    novelty_gap: float | None
    ci_low: float | None
    ci_high: float | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class GlobalNoveltySummary:
    strong_unseen_count: int
    protected_count: int
    suggestive_unseen_count: int
    possible_exposure_count: int
    broad_novel_mean_residual: float | None
    possible_exposure_mean_residual: float | None
    novelty_gap: float | None
    ci_low: float | None
    ci_high: float | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def classify_exposure(
    model: ModelDefinition,
    benchmark: BenchmarkDefinition,
    *,
    likely_unseen_days: int = 45,
) -> ExposureAssessment:
    """Classify how plausible benchmark-task exposure was before model release."""
    exposure_date = benchmark.public_since or benchmark.published_at
    if model.released_at is not None and exposure_date > model.released_at:
        return ExposureAssessment(
            ExposureTier.GUARANTEED_UNSEEN_POST_RELEASE,
            f"benchmark became public {exposure_date} after model release {model.released_at}",
        )
    if model.training_cutoff is not None and exposure_date > model.training_cutoff:
        return ExposureAssessment(
            ExposureTier.DISCLOSED_CUTOFF_UNSEEN,
            f"benchmark became public {exposure_date} after disclosed cutoff {model.training_cutoff}",
        )
    if benchmark.sealed_test:
        return ExposureAssessment(
            ExposureTier.SEALED_TEST,
            "test tasks are marked sealed/private; direct task contamination is unlikely",
        )
    if benchmark.rotating:
        return ExposureAssessment(
            ExposureTier.ROTATING_LOW_EXPOSURE,
            "benchmark is rotating; exact evaluation tasks have reduced exposure opportunity",
        )
    if model.released_at is not None:
        lead_days = (model.released_at - exposure_date).days
        if 0 <= lead_days <= likely_unseen_days:
            return ExposureAssessment(
                ExposureTier.LIKELY_UNSEEN_SHORT_LEAD,
                f"benchmark preceded release by only {lead_days} days; cutoff is not known to prove non-exposure",
            )
        if lead_days >= 0:
            return ExposureAssessment(
                ExposureTier.EXPOSURE_POSSIBLE,
                f"benchmark was public {lead_days} days before model release",
            )
    if model.training_cutoff is not None and exposure_date <= model.training_cutoff:
        return ExposureAssessment(
            ExposureTier.EXPOSURE_POSSIBLE,
            f"benchmark was public before disclosed cutoff {model.training_cutoff}",
        )
    return ExposureAssessment(ExposureTier.UNKNOWN, "insufficient release/cutoff metadata")


def _exposure_group(tier: ExposureTier) -> str:
    if tier in STRONG_UNSEEN:
        return "strong_unseen"
    if tier in PROTECTED:
        return "protected"
    if tier in SUGGESTIVE_UNSEEN:
        return "suggestive_unseen"
    if tier in POSSIBLY_EXPOSED:
        return "possible_exposure"
    return "unknown"


def _loading_similarity(left: BenchmarkDefinition, right: BenchmarkDefinition) -> float:
    a = np.asarray([left.capability_loadings.get(cap, 0.0) for cap in Capability], dtype=float)
    b = np.asarray([right.capability_loadings.get(cap, 0.0) for cap in Capability], dtype=float)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _aggregate_scores(
    benchmarks: dict[str, BenchmarkDefinition],
    observations: list[BenchmarkObservation],
) -> dict[tuple[str, str], float]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for observation in observations:
        benchmark = benchmarks.get(observation.benchmark_id)
        if benchmark is None:
            continue
        grouped[(observation.model_id, observation.benchmark_id)].append(
            normalize_score(observation.score, benchmark)
        )
    return {key: float(np.mean(values)) for key, values in grouped.items()}


def comparable_benchmark_residuals(
    models: list[ModelDefinition],
    benchmarks: list[BenchmarkDefinition],
    observations: list[BenchmarkObservation],
    *,
    minimum_overlap: int = 5,
    minimum_similarity: float = 0.20,
    minimum_correlation: float = 0.20,
    likely_unseen_days: int = 45,
) -> list[NoveltyResidual]:
    """Cross-predict each result from empirically related, capability-similar benchmarks.

    Each comparator is calibrated on other models that took both benchmarks, excluding
    the target model. This controls raw benchmark difficulty and prevents target leakage.
    """
    model_by_id = {model.id: model for model in models}
    benchmark_by_id = {benchmark.id: benchmark for benchmark in benchmarks}
    scores = _aggregate_scores(benchmark_by_id, observations)
    by_benchmark: dict[str, dict[str, float]] = defaultdict(dict)
    for (model_id, benchmark_id), score in scores.items():
        by_benchmark[benchmark_id][model_id] = score

    results: list[NoveltyResidual] = []
    for (model_id, target_id), observed in scores.items():
        model = model_by_id.get(model_id)
        target = benchmark_by_id.get(target_id)
        if model is None or target is None:
            continue
        predictions: list[tuple[float, float]] = []
        for comparator_id, comparator in benchmark_by_id.items():
            if comparator_id == target_id:
                continue
            model_comparator = scores.get((model_id, comparator_id))
            if model_comparator is None:
                continue
            similarity = _loading_similarity(target, comparator)
            if similarity < minimum_similarity:
                continue
            shared = sorted(
                (set(by_benchmark[target_id]) & set(by_benchmark[comparator_id])) - {model_id}
            )
            if len(shared) < minimum_overlap:
                continue
            x = np.asarray([by_benchmark[comparator_id][item] for item in shared], dtype=float)
            y = np.asarray([by_benchmark[target_id][item] for item in shared], dtype=float)
            if float(np.std(x)) < 1e-8 or float(np.std(y)) < 1e-8:
                continue
            corr = float(np.corrcoef(x, y)[0, 1])
            if not np.isfinite(corr) or corr < minimum_correlation:
                continue
            slope, intercept = np.polyfit(x, y, 1)
            prediction = float(np.clip(intercept + slope * model_comparator, 0.0, 100.0))
            weight = similarity * (corr**2) * math.log1p(len(shared))
            if weight > 0:
                predictions.append((prediction, weight))
        if not predictions:
            continue
        predicted = float(
            np.average(
                np.asarray([value for value, _ in predictions]),
                weights=np.asarray([weight for _, weight in predictions]),
            )
        )
        assessment = classify_exposure(model, target, likely_unseen_days=likely_unseen_days)
        results.append(
            NoveltyResidual(
                model_id=model_id,
                benchmark_id=target_id,
                observed_score=round(observed, 3),
                predicted_score=round(predicted, 3),
                residual=round(observed - predicted, 3),
                exposure_tier=assessment.tier,
                exposure_group=_exposure_group(assessment.tier),
                comparator_count=len(predictions),
                comparator_weight=round(sum(weight for _, weight in predictions), 4),
            )
        )
    return sorted(results, key=lambda row: (row.model_id, row.benchmark_id))


def _mean(values: list[float]) -> float | None:
    return None if not values else float(np.mean(values))


def _gap_interval(
    novel: list[float], exposed: list[float]
) -> tuple[float | None, float | None, float | None]:
    if not novel or not exposed:
        return None, None, None
    gap = float(np.mean(novel) - np.mean(exposed))
    if len(novel) < 2 or len(exposed) < 2:
        return gap, None, None
    novel_var = float(np.var(novel, ddof=1))
    exposed_var = float(np.var(exposed, ddof=1))
    se = math.sqrt(novel_var / len(novel) + exposed_var / len(exposed))
    return gap, gap - 1.96 * se, gap + 1.96 * se


def summarize_model_novelty(rows: list[NoveltyResidual]) -> list[ModelNoveltySummary]:
    by_model: dict[str, list[NoveltyResidual]] = defaultdict(list)
    for row in rows:
        by_model[row.model_id].append(row)
    summaries: list[ModelNoveltySummary] = []
    for model_id, model_rows in by_model.items():
        buckets: dict[str, list[float]] = defaultdict(list)
        for row in model_rows:
            buckets[row.exposure_group].append(row.residual)
        broad_novel = buckets["strong_unseen"] + buckets["protected"] + buckets["suggestive_unseen"]
        exposed = buckets["possible_exposure"]
        gap, low, high = _gap_interval(broad_novel, exposed)
        summaries.append(
            ModelNoveltySummary(
                model_id=model_id,
                strong_unseen_count=len(buckets["strong_unseen"]),
                protected_count=len(buckets["protected"]),
                suggestive_unseen_count=len(buckets["suggestive_unseen"]),
                possible_exposure_count=len(exposed),
                unknown_count=len(buckets["unknown"]),
                broad_novel_mean_residual=round(_mean(broad_novel), 3) if broad_novel else None,
                possible_exposure_mean_residual=round(_mean(exposed), 3) if exposed else None,
                novelty_gap=round(gap, 3) if gap is not None else None,
                ci_low=round(low, 3) if low is not None else None,
                ci_high=round(high, 3) if high is not None else None,
            )
        )
    return sorted(
        summaries,
        key=lambda row: (
            row.novelty_gap is None,
            row.novelty_gap if row.novelty_gap is not None else 0.0,
        ),
    )


def summarize_global_novelty(rows: list[NoveltyResidual]) -> GlobalNoveltySummary:
    buckets: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        buckets[row.exposure_group].append(row.residual)
    broad_novel = buckets["strong_unseen"] + buckets["protected"] + buckets["suggestive_unseen"]
    exposed = buckets["possible_exposure"]
    gap, low, high = _gap_interval(broad_novel, exposed)
    return GlobalNoveltySummary(
        strong_unseen_count=len(buckets["strong_unseen"]),
        protected_count=len(buckets["protected"]),
        suggestive_unseen_count=len(buckets["suggestive_unseen"]),
        possible_exposure_count=len(exposed),
        broad_novel_mean_residual=round(_mean(broad_novel), 3) if broad_novel else None,
        possible_exposure_mean_residual=round(_mean(exposed), 3) if exposed else None,
        novelty_gap=round(gap, 3) if gap is not None else None,
        ci_low=round(low, 3) if low is not None else None,
        ci_high=round(high, 3) if high is not None else None,
    )
