# Second density diagnostic — 2026-09-04

The second data pass was designed around **overlap density**, not benchmark count. The objective was to make the same frontier models recur across ordinary, agentic, protected and low-exposure evaluations so that latent capability and benchmark-sensitivity hypotheses could be tested separately.

## Corpus growth

The checked-in corpus now contains:

- **28 models**;
- **32 protocol-specific benchmark definitions**;
- **329 provenance-tagged observations**;
- raw model × benchmark density of approximately **36.7%**, up from 23.1% in the first seed.

The main density gain came from the 2026-06-25 LiveBench refresh: seven separate category measurements over a broad common model cohort. ARC Prize semi-private results add protected reasoning/spatial evidence, while a small MMMU-Pro snapshot begins to cover the visual domain.

The data layout is now modular. Historical root files coexist with immutable dated snapshots under `data/current/benchmarks/` and `data/current/observations/`.

## 1. A general factor exists, but it is not the whole model

A regularized rank-1 factor fit on the current sparse corpus retains a **24-model × 24-benchmark** core with **294/576 measured cells (51.0%)**.

The fit minimizes squared error only on observed cells. The earlier iterative missing-value SVD completion was rejected after it produced negative observed variance explained and nonsensical model ordering on the real block-missing matrix.

With observed-cell regularized factorization:

- rank 1 explains **41.4%** of observed standardized variance;
- rank 2 explains **62.0%** cumulatively;
- rank 3 explains **75.1%** cumulatively.

Factor 1 ranks the leading models plausibly in the current corpus:

| Model | Factor-1 score |
|---|---:|
| GPT-6 Astra | +2.293 |
| Claude Fable 5.1 | +1.588 |
| Claude Fable 5 | +1.537 |
| Claude Opus 5 | +1.034 |
| GPT-5.5 | +1.007 |
| GPT-5.6 Sol | +0.876 |
| Gemini 3.7 Flash | +0.634 |
| Kimi K3 | +0.308 |
| GPT-5.6 Terra | +0.215 |
| Qwen 3.8 Max | +0.149 |

The largest positive factor-1 benchmark loadings include HealthBench Professional, LiveBench Reasoning, Terminal-Bench 4.0, LiveBench Mathematics, LiveBench Language, ARC-AGI-2 semi-private, ARC-AGI-1 semi-private, ARC-AGI-3 Standard, and LiveBench Agentic Coding.

This is consistent with a meaningful `g`-like shared capability factor. But approximately 59% of observed variance remains outside rank 1, so treating intelligence as one scalar would discard most of the measured structure.

### Complete LiveBench block cross-check

The fully observed 22-model × 7-category LiveBench block provides a useful check without missing-data modeling. Ordinary PCA gives approximately:

- PC1: **43.4%**;
- PC2: **20.9%**;
- PC3: **12.4%**.

PC1 loads positively across all seven categories. PC2 separates instruction-following/language behavior from coding/data-analysis behavior. This independently supports the general-factor-plus-domain-residual interpretation.

## 2. Terminal performance is more than ordinary coding

Across the eleven models shared with the current Terminal-Bench 4.0 sample:

| Pair | Pearson | Spearman |
|---|---:|---:|
| TB4 vs LiveBench Coding | 0.608 | 0.620 |
| TB4 vs LiveBench Agentic Coding | **0.742** | **0.638** |

The rank difference is modest, but score amplitude tracks the agentic-coding category materially better than ordinary coding.

This strengthens the original interpretation of the DeepSWE / Terminal-Bench discrepancy: a model can be a strong repository engineer while being much weaker at terminal/environment interaction. That is a capability-profile difference, not automatically evidence of contaminated coding scores.

## 3. New benchmarks can create a new capability regime

ARC-AGI-2 semi-private versus ARC-AGI-3 Standard has only seven common models in the current seed, so this remains descriptive. Nevertheless:

- Pearson is approximately **0.651**;
- Spearman is approximately **0.893**.

The broad ordering transfers surprisingly well, while score magnitude changes dramatically. GPT-5.6 Sol and Terra fall roughly twenty points below a simple ARC-2 → ARC-3 amplitude prediction, while GPT-6 Astra substantially exceeds it.

This is an important confound for the benchmaxxing hypothesis. A newly released benchmark can probe a genuinely different regime and therefore produce large negative residuals without contamination or benchmark-specific overfitting being the cause.

The final model therefore needs domain/capability interactions and cannot interpret every new-benchmark drop as an exposure effect.

## 4. The broad benchmaxxing hypothesis does not currently reproduce

The expanded novelty detector yields **263 usable leave-target-out residuals**:

- strong/provably unseen: 47;
- protected: 35;
- short-lead/suggestive unseen: 62;
- possible exposure: 35;
- remaining usable residuals fall in other/unknown exposure categories.

The descriptive global novelty gap is:

**−0.09 points**

with a naive 95% interval of approximately:

**−1.95 to +1.76**

This interval is not a valid clustered inferential interval and must not be reported as a formal hypothesis test. However, the point estimate is already useful: the present corpus gives **no evidence of a frontier-wide collapse when benchmark exposure is reduced**.

That falsifies the strongest version of the internet claim. Frontier benchmark leaders do not, as a class, appear unable to generalize to new or protected measurements.

## 5. Gemini 3.8 Flash is still a large model-specific anomaly

The aggregate result does not erase the motivating observation.

For Gemini 3.8 Flash, the current V0 residual analysis finds approximately:

- low-exposure/protected mean residual: **−20.23**;
- possibly exposed mean residual: **+0.89**;
- novelty gap: approximately **−21.12**.

Its Terminal-Bench 4.0 result is the direct contributor we expected:

- observed: **19.1**;
- current cross-benchmark prediction: approximately **39.3**;
- residual: approximately **−20.2**.

This is a strong anomaly, but **not proof of contamination**. Plausible explanations still include:

1. a real terminal/environment-agency weakness;
2. harness/tool interaction;
3. capability-loadings misspecification;
4. benchmark-family nonlinearity;
5. limited comparator overlap;
6. benchmark-specific post-training differences;
7. contamination or direct benchmark optimization.

The contamination/benchmaxxing interpretation becomes persuasive only if Gemini 3.8 shows the same directional residual on several independent, low-exposure benchmark families that measure capabilities it otherwise appears to possess.

## 6. Negative controls matter

The expanded dataset also contains models that perform strongly on evaluations released after or outside their likely optimization window. In particular, older Claude revisions remain highly competitive on the June LiveBench refresh.

These cases are important because they demonstrate that strong performance on historical benchmarks does **not** mechanically imply collapse on a newly refreshed evaluation. Any eventual benchmark-sensitivity parameter must therefore be model-specific, not a universal age penalty applied to frontier models.

## Provisional conclusion

The project has moved from a plausible idea to a testable statistical problem.

The current evidence supports three claims:

1. **A substantial general capability factor exists.** Rank 1 explains roughly 41–43% of variance in two different diagnostics.
2. **Capability remains strongly multidimensional.** Most variance is not captured by the general factor, and agentic/terminal behavior is distinguishable from ordinary coding.
3. **There is no current global benchmaxxing effect, but there are large model-specific novelty anomalies.** Gemini 3.8 Flash is presently the clearest motivating example.

The next estimator should therefore model:

`performance = benchmark difficulty + g + domain capability + protocol/harness + model × exposure effect + error`

The quantity of interest for the benchmaxxing hypothesis is the **model-specific exposure interaction**, not raw benchmark age and not the global average drop.

## Next data problem: model identity is time-dependent

The exposure analysis assumes that a model ID identifies fixed weights. That is not always true. API providers can update a model while keeping the same public name. DeepSeek's public changelog, for example, explicitly states that the August 2026 DeepSeek-V4-Pro GA update kept the same API model name.

This means `model_id = deepseek-v4-pro` is not sufficient provenance for longitudinal evaluation. Better Bench needs revision-aware model identity before using release dates as causal exposure metadata. The observation schema should therefore evolve toward explicit model revision/version metadata, with immutable revision IDs wherever providers expose them.
