# Better Bench Index — 2026-09-04 snapshot

Calibration: `BBI-2026-09-04`

The Better Bench Index (BBI) is a presentation transform of the validated fixed-scale
general-capability estimator:

`BBI = 100 + 10 * general_z`

For this calibration, 100 is the retained frontier-cohort mean and 10 index points equal
one latent standard deviation. BBI is versioned because the reference cohort and evidence
graph evolve.

Uncertainty combines the estimator's conditional standard error with fixed-calibration
leave-one-benchmark-family-out sensitivity. For each model, one independent benchmark
family is removed at a time and the model is re-projected onto the already learned
benchmark intercepts/loadings. This measures dependence on any one family without letting
the held-out family re-calibrate the scale.

Current model×family cross-validation: 500 held-out observations; model RMSE 9.076
normalized benchmark points versus 15.938 for the benchmark-mean baseline, a 43.05% RMSE
reduction; held-out residual R² 0.6757 and Spearman 0.7338.

| Rank | Model | BBI | 95% interval | Variance | Families | Coverage | Confidence |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | GPT-6 Astra | 119.9 | 115.4–124.5 | 5.42 | 9 | 28.1% | established |
| 2 | Claude Fable 5.1 | 116.7 | 114.2–119.2 | 1.61 | 15 | 46.9% | mature |
| 3 | Claude Opus 5 | 112.5 | 110.7–114.2 | 0.81 | 17 | 53.1% | mature |
| 4 | Muse Spark 1.3 | 110.0 | 105.2–114.8 | 6.08 | 5 | 15.6% | fresh |
| 5 | Claude Fable 5 | 109.2 | 107.2–111.3 | 1.12 | 20 | 62.5% | mature |
| 6 | GPT-5.6 Sol | 108.8 | 107.4–110.3 | 0.52 | 22 | 68.8% | mature |
| 7 | GLM-5.3 | 108.3 | 104.7–111.9 | 3.34 | 7 | 21.9% | fresh |
| 8 | Kimi K3 | 106.8 | 104.4–109.2 | 1.56 | 11 | 34.4% | established |
| 9 | Grok 4.6 | 106.3 | 103.9–108.7 | 1.47 | 9 | 28.1% | established |
| 10 | Gemini 3.8 Flash | 105.7 | 102.6–108.8 | 2.50 | 8 | 25.0% | established |
| 11 | Qwen 3.8 Max | 105.3 | 102.5–108.1 | 1.99 | 9 | 28.1% | established |
| 12 | GPT-5.6 Terra | 104.8 | 102.9–106.7 | 0.93 | 12 | 37.5% | mature |
| 13 | Claude Opus 4.8 | 104.5 | 102.7–106.4 | 0.88 | 16 | 50.0% | mature |
| 14 | Gemini 3.7 Flash | 103.6 | 100.5–106.6 | 2.41 | 9 | 28.1% | established |
| 15 | Muse Spark 1.2 | 103.4 | 101.1–105.7 | 1.41 | 9 | 28.1% | established |

## Kimi K3 evidence-depth change

Before the 2026-09-04 Kimi depth pass, Kimi K3 had 6 retained independent benchmark
families, `general_z = +0.4413`, and ranked 14th in the point estimator. The depth pass
added direct/independent evidence from FrontierSWE v2, MLS-Bench Lite, MCPMark Verified,
Agents' Last Exam and AA-Briefcase. After refitting, Kimi has 11 retained families,
`general_z = +0.6801`, BBI 106.8, and ranks 8th. Its effective family evidence is 9.631.

The added evidence is not uniformly favorable: Kimi leads the stored MCPMark Verified
snapshot at 96.06% and is strong on AA-Briefcase (1543 Elo) and MLS-Bench Lite (48.3%),
while FrontierSWE v2 is a counterweight at 25.9%, behind Claude Fable 5.1, GPT-5.6 Sol
and GLM-5.3. This is useful because the rank improvement comes from broader evidence,
not from selecting only benchmarks on which Kimi wins.

## Interpretation caveat

The interval is a current-data uncertainty interval, not yet a full Bayesian posterior.
It includes conditional estimator uncertainty and benchmark-family sensitivity, but not
all sources of uncertainty such as per-benchmark sampling error, calibration-parameter
uncertainty, model revision ambiguity, reasoning-effort mismatch, or missing-not-at-random
leaderboard selection. Those should be incorporated in later estimator versions.
