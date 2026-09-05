from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import minimize

from .consensus_ranking import (
    ConsensusRankingConfig,
    PairPreference,
    build_pair_preferences,
)


@dataclass(frozen=True)
class BradleyTerryConfig:
    margin_temperature_points: float = 5.0
    minimum_shared_families: int = 1
    ridge: float = 0.3
    use_learned_discrimination: bool = False
    maximum_iterations: int = 500


@dataclass(frozen=True)
class BradleyTerryResult:
    scores: dict[str, float]
    ranking: list[str]
    rank: dict[str, int]
    preferences: list[PairPreference]
    converged: bool
    iterations: int
    objective: float


def fit_bradley_terry(
    rows: list[Any],
    model_ids: list[str],
    state: Any | None,
    *,
    score_unit_points: float,
    config: BradleyTerryConfig | None = None,
) -> BradleyTerryResult:
    """Fit a regularized Bradley-Terry model to incomplete benchmark comparisons.

    Each aggregated pair preference becomes a fractional binomial observation. If its
    total comparison information is ``W`` and signed preference is ``D``, the implied
    left-win mass is ``(W + D)/2`` and right-win mass ``(W - D)/2``. This preserves
    benchmark quality/provenance weighting and bounded score margins while producing a
    globally coherent continuous capability scale.

    Missing benchmark cells create no event. They therefore cannot improve a model's
    score merely by being absent; sparse comparison connectivity is handled by ridge
    shrinkage toward the population mean.
    """

    config = config or BradleyTerryConfig()
    if config.ridge <= 0:
        raise ValueError("ridge must be positive")
    if not model_ids:
        raise ValueError("model_ids cannot be empty")

    preference_config = ConsensusRankingConfig(
        margin_temperature_points=config.margin_temperature_points,
        minimum_shared_families=config.minimum_shared_families,
        use_learned_discrimination=config.use_learned_discrimination,
    )
    preferences = build_pair_preferences(
        rows,
        model_ids,
        state,
        score_unit_points=score_unit_points,
        config=preference_config,
    )
    index = {model_id: position for position, model_id in enumerate(model_ids)}

    events: list[tuple[int, int, float, float]] = []
    for preference in preferences:
        total = float(preference.total_information)
        if total <= 1e-12:
            continue
        signed = float(np.clip(preference.net_preference, -total, total))
        target = float(np.clip((total + signed) / (2.0 * total), 1e-8, 1.0 - 1e-8))
        events.append(
            (
                index[preference.left_model],
                index[preference.right_model],
                target,
                total,
            )
        )
    if not events:
        raise ValueError("No pairwise comparison events available")

    def loss_and_gradient(raw: np.ndarray) -> tuple[float, np.ndarray]:
        centered = raw - float(raw.mean())
        loss = 0.5 * config.ridge * float(np.dot(centered, centered))
        gradient = config.ridge * centered
        for left, right, target, weight in events:
            difference = float(centered[left] - centered[right])
            # Stable logistic and cross entropy.
            if difference >= 0:
                exp_neg = np.exp(-difference)
                probability = 1.0 / (1.0 + exp_neg)
                log_probability = -np.log1p(exp_neg)
                log_one_minus = -difference - np.log1p(exp_neg)
            else:
                exp_pos = np.exp(difference)
                probability = exp_pos / (1.0 + exp_pos)
                log_probability = difference - np.log1p(exp_pos)
                log_one_minus = -np.log1p(exp_pos)
            loss -= weight * (
                target * log_probability + (1.0 - target) * log_one_minus
            )
            derivative = weight * (probability - target)
            gradient[left] += derivative
            gradient[right] -= derivative
        gradient -= float(gradient.mean())
        return float(loss), gradient

    result = minimize(
        fun=lambda x: loss_and_gradient(x)[0],
        x0=np.zeros(len(model_ids), dtype=float),
        jac=lambda x: loss_and_gradient(x)[1],
        method="L-BFGS-B",
        options={"maxiter": config.maximum_iterations, "ftol": 1e-12, "gtol": 1e-8},
    )
    raw = np.asarray(result.x, dtype=float)
    raw -= float(raw.mean())
    scale = float(raw.std(ddof=0))
    if scale <= 1e-12:
        scale = 1.0
    standardized = raw / scale
    scores = {
        model_id: float(standardized[index[model_id]]) for model_id in model_ids
    }
    ranking = sorted(model_ids, key=lambda model_id: scores[model_id], reverse=True)
    ranks = {model_id: rank for rank, model_id in enumerate(ranking, start=1)}
    return BradleyTerryResult(
        scores=scores,
        ranking=ranking,
        rank=ranks,
        preferences=preferences,
        converged=bool(result.success),
        iterations=int(result.nit),
        objective=float(result.fun),
    )
