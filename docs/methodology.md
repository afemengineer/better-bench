# Methodology

## Objective

Better Bench aims to recover the best available **general-intelligence ordering** of frontier models from heterogeneous, incomplete and non-random benchmark evidence. Benchmark scores are treated as noisy measurements with protocol, provenance, contamination and family structure; missing evaluations must never become an implicit advantage.

The production methodology deliberately separates two statistical jobs that should not be conflated:

1. **Calibration and admission** — determine whether a model has enough independent evidence to enter the official table and learn benchmark difficulty/discrimination where the matrix is sufficiently dense.
2. **Official ordering** — rank admitted models using coverage-neutral comparisons on the broadest defensible evidence panel.

This separation was introduced after family-held-out diagnostics showed that a single-factor score could reverse well-overlapped direct comparisons when models had systematically different benchmark portfolios.

## Evidence integrity

Every observation retains benchmark version, model identity/revision, source, harness, reasoning effort and other protocol metadata where available. Mutable product aliases are not treated as one immutable statistical object across weight updates: post-cutover observations that cannot be pinned to a defensible revision remain auditable raw evidence but do not enter scoring.

Benchmark evidence is weighted by scientific quality rather than raw leaderboard count. Quality incorporates protocol quality, reliability, contamination resistance, independence/redundancy and a deliberately modest adoption signal. Related protocols share a benchmark-family budget so duplicated or closely related leaderboards cannot acquire multiple independent votes.

Source provenance is also weighted. Independent benchmark-owner or high-quality third-party measurements receive more influence than provider-run launch tables.

## Stage A — calibration and rankability

The fixed one-factor estimator is retained because it predicts held-out benchmark scores substantially better than benchmark means and provides useful cardinal calibration.

For normalized score `y_mb`, model `m` and benchmark `b`, its core form is approximately:

`y_mb = alpha_b + lambda_b * g_m + error_mb`

where:

- `alpha_b` captures benchmark difficulty;
- `lambda_b` captures frontier discrimination;
- `g_m` is a calibration latent score;
- evidence weights incorporate benchmark family and provenance quality.

The calibration panel requires enough overlapping models to estimate benchmark parameters reliably. Models also require a minimum number of independent benchmark families before becoming officially rankable. Models below that gate remain provisional.

The resulting `g_m` and the familiar `100 + 10*g_m` BBI transform are now treated as **calibration magnitude diagnostics**, not the official sorting key.

## Why calibration score no longer determines rank

Benchmark coverage is not missing at random. New models are frequently evaluated on fashionable, convenient or vendor-selected subsets. Difficult evaluations may arrive later or never be published. A model with fewer hard benchmark cells can therefore obtain a higher single-factor point estimate even when matched evidence favors another model.

This is unacceptable for a general-intelligence classification: absence of a hard evaluation must increase uncertainty, not improve rank.

## Stage B — broad coverage-neutral ranking panel

Once a model passes the conservative rankability gate, final ordering uses a broader panel. A benchmark does **not** need five models to contribute to a head-to-head comparison; it needs only two already-rankable models because benchmark difficulty cancels within that comparison.

The ranking panel therefore includes every revision-safe benchmark with results for at least two rankable models. In the current snapshot this expands the usable ordering evidence beyond the dense calibration matrix, including high-quality emerging benchmarks that are especially informative at the frontier.

Ranking weights are intentionally **exogenous to the current leaderboard**. They use benchmark quality, integrity, independence, a weak adoption term and source reliability. Learned single-factor discrimination is not used in the final ordering weights; otherwise the old ranking can leak back into the correction.

For models `i` and `k` sharing benchmark `j`, comparison information uses the profiled weighted-least-squares form:

`I_ikj proportional to w_ij * w_kj / sum_m(w_mj)`

This prevents a benchmark evaluated on many models from gaining quadratic influence merely because it generates many model pairs.

Raw score margins are bounded with a smooth `tanh` transform. Large wins remain stronger evidence than tiny wins, but a single numerically wide metric cannot overwhelm many independent benchmark families.

## Stage C — weighted Kemeny consensus

The official ordering is `broad-kemeny-v1`.

Each benchmark family supplies a partial weighted ordering over the models it actually evaluated. Missing cells create **no comparison and no positive evidence**. Better Bench then solves a weighted Kemeny rank-aggregation problem: find the globally transitive total order that minimizes weighted disagreement with those incomplete pairwise preferences.

This is solved as a binary mixed-integer program with triangle constraints enforcing transitivity. It directly optimizes the object Better Bench ultimately publishes — model ordering — rather than treating ordinal accuracy as a side effect of score regression.

The current deployment configuration is:

- margin temperature: 10 normalized score points;
- minimum direct shared families: 1;
- learned discrimination in ranking weights: disabled;
- conservative fixed-estimator rankability gate retained.

No model-specific ordering rule is used.

## Validation

Method selection is evaluated by holding out **entire benchmark families**, not random rows. This is intentionally difficult: the ranking must generalize to a new evaluation family rather than interpolate another variant of a benchmark it already saw.

Hyperparameters were also tested with nested family cross-validation so the reported improvement is not simply the result of selecting a configuration on the same folds used for evaluation.

On the current 33-model frontier cohort, nested whole-family validation gives the following comparison against the former fixed-score ordering:

| Held-out ordering target | Fixed calibration order | Broad Kemeny |
| --- | ---: | ---: |
| All non-tied pairs | 71.67% | 72.43% |
| >1 point gaps | 73.43% | 74.54% |
| >3 point gaps | 76.78% | 77.78% |
| >5 point gaps | 79.47% | 80.43% |
| >10 point gaps | 84.82% | 86.67% |
| Frontier non-tied pairs | 67.58% | 73.44% |
| Frontier >3 point gaps | 75.00% | 82.35% |
| Frontier >5 point gaps | 79.67% | 86.59% |
| Frontier >10 point gaps | 85.21% | 94.37% |

The gains are largest where Better Bench is intended to be strongest: distinguishing current frontier models on previously unseen benchmark families.

## Rank uncertainty and stability

A single integer rank still overstates certainty when models are close. Official outputs therefore include a leave-one-benchmark-family-out **rank stability band**. For every ranking family, Better Bench removes that family, refits the entire consensus order, and records each model's minimum and maximum rank.

These are robustness intervals, not frequentist confidence intervals. They answer a concrete sensitivity question: how far can this model move if any one benchmark family is removed?

The current full ranking has median leave-one-family-out Spearman correlation of approximately `0.9997`, minimum approximately `0.9659`, and median top-10 overlap of `100%`.

Direct pairwise preferences are also audited. Kemeny may occasionally violate a weak direct preference to maintain the globally best transitive consensus, but the current solution agrees with approximately `98.6%` of weighted direct-preference mass.

## Numeric BBI versus official rank

The numeric BBI value remains useful as a calibrated magnitude diagnostic, with its existing conditional and family-sensitivity uncertainty. It is **not** currently claimed to be a coverage-neutral cardinal intelligence interval.

Therefore production output exposes both:

- `rank`, `rank_low`, `rank_high`, `ranking_method` — the official general-intelligence classification;
- calibration BBI / `general_z` and its interval — auxiliary cardinal calibration information.

The official table is sorted by consensus rank, never by calibration BBI.

A future continuous headline score should only replace this separation if it matches or exceeds broad Kemeny on unseen-family ordering while preserving calibrated cardinal interpretation.

## Capability profiles

Domain residuals and capability-taxonomy views still use the calibrated latent model. They are diagnostic profiles rather than independent official general-intelligence rankings. This keeps software engineering, reasoning, tool use, long context and other capability information visible without silently double-counting those dimensions in the headline order.

## Model versus system capability

Agentic benchmarks often measure `model + harness + tools + budget`, not the model in isolation. Observations retain harness, reasoning effort, token budget and provenance. Protocol-specific results are not silently merged when the scaffold materially changes the measured system.
