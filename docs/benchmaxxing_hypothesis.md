# Benchmaxxing / novelty-robustness hypothesis

Better Bench has two goals: estimate latent capability, and detect when apparent capability depends unusually strongly on benchmark exposure.

## Hypothesis

A genuinely capable model should transfer reasonably well to a newly released or otherwise protected benchmark that measures capabilities it already demonstrates elsewhere. A model whose public benchmark strength is partly driven by contamination, benchmark-specific post-training, or over-optimization should show systematically negative residuals when evaluated on benchmarks it could not plausibly have seen.

This is an **interaction hypothesis**, not a raw-score hypothesis. New benchmarks are often harder by design, so `new score < old score` is not evidence by itself.

Better Bench instead asks:

> Given the model's results on empirically related benchmarks, and given how other models map between those benchmarks, did this model underperform its expected result specifically when exposure opportunity was low?

## Exposure evidence tiers

1. **Guaranteed unseen — post release**: benchmark tasks became public only after the model version was released.
2. **Disclosed-cutoff unseen**: benchmark tasks became public after a vendor-disclosed training cutoff.
3. **Sealed test**: evaluation tasks are private/held out. This reduces direct task contamination but is not identical to a post-training benchmark.
4. **Rotating / low exposure**: exact evaluation content changes, reducing memorization opportunity.
5. **Likely unseen — short lead**: benchmark became public shortly before model release, but no cutoff proves non-exposure.
6. **Exposure possible**: benchmark was public sufficiently early that contamination or targeted post-training cannot be excluded.
7. **Unknown**: metadata is insufficient.

A short release lead is never promoted to "guaranteed unseen". Terminal-Bench 4.0 currently predates Gemini 3.8 Flash's release by only days; without a disclosed Gemini training cutoff that is suggestive evidence, not proof.

## V0 residual estimator

For each observed model `m` on target benchmark `b`:

1. Find other benchmarks `c` that the same model took.
2. Require `b` and `c` to have enough other models in common.
3. Exclude model `m` from the calibration set.
4. Fit the empirical mapping `score_b ~ score_c` using the other shared models.
5. Weight that prediction by capability-loading similarity, empirical cross-model correlation, and overlap size.
6. Combine comparator predictions into an expected target score.
7. Compute `novelty residual = observed target score - predicted target score`.

A negative residual means the model did worse than its performance on comparable benchmarks predicted. Because the mapping is calibrated on the target benchmark itself using other models, raw benchmark difficulty and scale are largely controlled.

## Model-level novelty gap

V0 reports:

`novelty gap = mean residual on low-exposure/new/protected targets - mean residual on possibly exposed targets`

A materially negative gap is the expected signature of benchmark sensitivity. It is **not** automatically called contamination. Other explanations include harness mismatch, a real missing sub-capability, poor taxonomy loadings, model-specific tool incompatibility, small cohorts, and evaluation noise.

## What would count as persuasive evidence?

A single poor result is not enough. A convincing benchmaxxing claim should eventually require several low-exposure or provably unseen benchmarks, more than one benchmark family/evaluator, capability-matched comparisons, consistent negative residuals, uncertainty intervals excluding a trivial effect, controls for harness/reasoning/token budget, and sensitivity checks under different taxonomies and residual estimators.

## V1 statistical model

The end-state should estimate capability and benchmark sensitivity jointly:

`y_mbp = benchmark_difficulty_b + general_capability_m + capability_vector_m · benchmark_loadings_b + harness_effect_p + exposure_effect_m * low_exposure_mb + error_mbp`

`exposure_effect_m` is the model-specific quantity of interest. A strongly negative value, with enough protected/post-training evidence, would mean that measured capability generalizes unusually poorly when benchmark exposure is removed.

This can later become hierarchical Bayesian / multidimensional IRT, with benchmark age and contamination probability modeled continuously instead of through bins.

## Why this belongs inside the intelligence benchmark

Novelty robustness should not replace capability. A model can be excellent and benchmark-sensitive, mediocre and robust, or excellent and robust.

Better Bench should expose at least three outputs:

1. **Capability profile / general capability estimate**.
2. **Novelty robustness** — transfer to protected or post-training measurements.
3. **Evidence coverage** — how much direct evidence supports both conclusions.

The ultimate ranking can weight low-contamination evidence more heavily, while keeping novelty robustness visible instead of silently folding it into one number.
