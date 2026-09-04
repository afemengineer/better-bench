# Methodology

## Objective

Better Bench treats benchmark scores as noisy observations of latent AI capabilities rather than as interchangeable points in a leaderboard. The primary artifact is a capability profile with uncertainty and evidence coverage. A single headline score is a derived view.

## V0 model

For an observation of model `m` on benchmark `b`, Better Bench first maps the raw result to a fixed benchmark scale. It never divides by the current best model, because doing so would make historical model scores change when a new frontier model is released.

Each observation receives an evidence weight:

`source × protocol × reliability × freshness × contamination × redundancy`

The benchmark's capability-loading vector then distributes that evidence across latent domains. For example, a repository engineering benchmark can primarily measure software engineering while also loading on planning and terminal/tool use.

### Freshness

Age is a mild prior rather than a hard rule. Old sealed or rotating benchmarks can remain useful.

### Contamination

Contamination is benchmark-model conditional. A public benchmark cannot contaminate a model whose training cutoff predates its release. Static public tests gradually receive lower weight when they have been exposed before a model's likely training window. This V0 term is explicitly heuristic. It represents risk, not an accusation that any model trained on test data.

### Redundancy

Where enough overlapping model observations exist, highly correlated benchmarks are down-weighted so that a domain does not become important merely because it has many near-duplicate leaderboards.

### Missing data

Missing benchmarks are never zeros. A capability with no direct evidence is reported as missing. The headline score re-normalizes over observed capability domains and its uncertainty interval widens as coverage falls. This makes the distinction between "measured weak" and "not measured" explicit.

## Why V0 is not the end state

Weighted means cannot fully solve cross-benchmark calibration or infer a genuinely multidimensional latent ability vector from sparse data. Once the result matrix contains enough models and benchmark items, the intended V1 estimator is a hierarchical Bayesian factor/IRT model:

`y_mb ~ f_b(lambda_b^T theta_m, benchmark difficulty, protocol variance)`

where `theta_m` is the latent capability vector and `lambda_b` contains benchmark loadings. Item-level IRT should be used when item responses are public; aggregate-score likelihoods can coexist for leaderboards that only publish summary results.

V1 should additionally model:

- model × harness effects for agentic evaluations;
- reasoning-effort and token-budget effects;
- source-specific uncertainty;
- benchmark version drift;
- posterior pairwise rank probabilities instead of pretending close point estimates are ordered;
- learned benchmark loadings, compared against the hand-authored taxonomy;
- benchmark information/discrimination at the current frontier.

## Model vs system capability

Agentic benchmarks often measure `model + harness + tools + budget`, not the model in isolation. Observations therefore retain harness, reasoning effort, token budget and provenance. Future views should expose both standardized model capability and best-known system capability rather than mix them silently.
