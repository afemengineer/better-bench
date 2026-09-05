# First real-data diagnostic — 2026-09-04

The initial public-data seed contains **28 models, 21 protocol-specific benchmarks and 136 observations**.

This is enough to test the premise, but not enough to publish a defensible one-number intelligence ranking.

## 1. Missingness is the first statistical problem

The model × benchmark matrix is only **23.1% observed**.

Across the 210 possible benchmark pairs:

- 89 pairs share at least 4 models;
- 60 share at least 5;
- only 20 share at least 6;
- only 2 share at least 8;
- only 1 shares at least 10.

Therefore a conventional correlation matrix or factor analysis would currently be dominated by tiny, source-specific cohorts.

This validates the original design decision: missing benchmarks must widen uncertainty rather than become zeroes or disappear inside a naive average.

## 2. "Coding" is visibly not one variable

The cleanest current overlap is DeepSWE v1.1 vs Terminal-Bench 4.0:

- overlap: **11 models**
- Pearson correlation: **0.572**
- Spearman rank correlation: **0.570**

That is substantial shared signal, but much too weak to call the two benchmarks interchangeable.

A linear residual check is more revealing. Given each model's DeepSWE score, the largest Terminal-Bench deviations are:

| Model | DeepSWE | TB 4.0 | TB predicted from DeepSWE | Residual |
|---|---:|---:|---:|---:|
| Gemini 3.8 Flash | 74.0 | 19.1 | 43.6 | **-24.5** |
| GPT-5.6 Luna | 67.0 | 17.3 | 33.2 | **-15.9** |
| Grok 4.6 | 67.0 | 20.3 | 33.2 | **-12.9** |
| GPT-6 Astra | 74.0 | 57.9 | 43.6 | **+14.3** |
| Claude Fable 5.1 | 67.4 | 55.8 | 33.8 | **+22.0** |

So the discrepancy that motivated Better Bench is not anecdotal. Repository engineering and terminal/environment agency share a factor but also produce large model-specific residuals.

## 3. Terminal agency appears to transfer into scientific workflows

Terminal-Bench 4.0 vs Terminal-Bench-Science 0.1 currently has:

- overlap: **6 models**
- Pearson: **0.867**
- Spearman: **0.943**

The sample is small, but this is exactly what the hand taxonomy predicts: both benchmarks load materially on terminal and planning agency even though one is "coding/ops" and the other is "science".

## 4. Some nominally similar coding benchmarks disagree sharply

Within the six-model current launch cohort:

- FrontierCode 1.1 Main vs Extended: Spearman **0.899**
- DeepSWE vs FrontierCode Main: Spearman **-0.152**
- DeepSWE vs FrontierCode Extended: Spearman **-0.277**

This is not yet evidence that DeepSWE and FrontierCode measure opposite abilities. Six highly selected frontier models, narrow score ranges, harness differences and vendor-reported FrontierCode values make that inference unsafe.

But it *is* evidence that averaging all three into a single "coding" column without modeling protocol and uncertainty would be statistically indefensible.

## 5. The hand taxonomy does not yet validate cleanly from raw pair correlations

For benchmark pairs with at least four common models, the Spearman correlation between:

1. cosine similarity of our hand-authored capability loadings, and
2. absolute observed benchmark rank correlation

is only about **0.11**.

This does **not** currently falsify the taxonomy. Raw benchmark correlations are a poor validation target because:

- general model quality creates positive cross-domain correlation ("g");
- most pairwise overlaps contain only four to six models;
- many pairs come from the same source-specific cohort;
- several frontier benchmarks are saturated;
- agent harnesses are mixed with base-model capability.

The next statistical version should estimate a general factor first, then ask whether residual covariance is explained by the proposed sub-capabilities. That is a hierarchical factor/IRT problem, not a correlation-clustering problem.

## Provisional verdict

**The project has signal. The simple scoring model does not yet have enough data to deserve a public headline rank.**

The strongest result so far is the DeepSWE / Terminal-Bench split: a shared coding/agentic component exists, while model-specific deviations are large enough that a multidimensional profile is materially more informative than a coding average.

The next data milestone should maximize *overlap*, not raw benchmark count: add public results for the same 15–25 models across visual/spatial, computer-use, long-context, knowledge and reasoning benchmarks. A dense 15-model × 20-benchmark core is more valuable for latent-factor estimation than a sparse 100-benchmark catalog.
