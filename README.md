# Better Bench

**Better Bench** is an open framework for answering a harder question than “which model has the highest average benchmark score?”

> What latent capabilities does a model actually demonstrate, how strong is the evidence, how well does that capability transfer to protected/new measurements, and how certain are we when the public benchmark matrix is sparse, redundant and potentially contaminated?

The project is deliberately built around **capability profiles first, leaderboard second**.

## V0 principles

- **Multidimensional intelligence.** Benchmarks load onto several capabilities instead of one bucket.
- **Missing ≠ zero.** Unevaluated domains remain missing and reduce evidence coverage.
- **Uncertainty is first-class.** Headline scores ship with intervals and effective evidence counts.
- **Fixed calibration.** Scores use benchmark-defined anchors, never “divide by today's best model.”
- **Contamination is conditional.** Public exposure is evaluated against the model's training/release window.
- **Novelty robustness is visible.** New/protected benchmark underperformance is measured as a residual, not hidden inside the headline score.
- **Benchmark importance is explicit.** Popularity matters, but cannot rescue a saturated, unreliable, low-integrity or redundant benchmark.
- **Benchmark families share an evidence budget.** Subdividing one suite into many columns does not create many independent votes.
- **Redundancy is penalized.** Highly correlated benchmarks should not receive ten votes for one skill.
- **Provenance matters.** Independent evaluations weigh more than secondary or unclear reports.
- **Model ≠ agent harness.** Harness, reasoning effort and token budget are retained in every observation.

The current estimator is intentionally transparent. It is scaffolding for a later hierarchical Bayesian / multidimensional IRT model, not a claim that weighted averages solve psychometrics. See [`docs/methodology.md`](docs/methodology.md), [`docs/benchmaxxing_hypothesis.md`](docs/benchmaxxing_hypothesis.md), and [`docs/benchmark_importance.md`](docs/benchmark_importance.md).

## Capability taxonomy

V0 tracks 15 domains: fluid reasoning, quantitative reasoning, scientific reasoning, knowledge, language, software engineering, structured tool use, terminal agency, web agency, GUI/computer use, visual intelligence, spatial intelligence, long-context/state tracking, planning/agency, and social/pragmatic intelligence.

The taxonomy is configured in [`config/taxonomy.yaml`](config/taxonomy.yaml) and is expected to evolve as empirical benchmark correlations and factor analysis give us better evidence.

## Quick start

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -e ".[dev]"

better-bench score \
  --benchmarks data/examples/benchmarks.yaml \
  --models data/examples/models.yaml \
  --observations data/examples/observations.csv \
  --as-of 2026-09-04 \
  --output-json scores.json
```

The synthetic example remains useful for tests. The real public-data corpus in [`data/current/`](data/current/) now contains **40 registered models, 41 protocol-specific benchmark definitions and 414 provenance-tagged observations**.

The full 40 × 41 registry is intentionally sparse (~25.2% raw cell density) because the latest expansion added twelve models only where credible public benchmark evidence exists. **Registry size is not the same thing as ranking eligibility.** The current structural factor diagnostic requires at least five benchmark observations per retained model, so only the denser subset enters that analysis until the new models accumulate enough independent evidence.

`data/current/` is modular: historical root files coexist with immutable dated snapshots under `models/`, `benchmarks/`, `observations/`, and `adoption/`. Passing the directory to the CLI loads the full corpus. Adoption is stored as dated evidence rather than permanent benchmark metadata.

Inspect empirical benchmark relationships:

```bash
better-bench diagnose \
  --benchmarks data/current \
  --observations data/current \
  --minimum-overlap 4 \
  --left deepswe-v1.1 \
  --right terminal-bench-4.0
```

Test the benchmaxxing / novelty-robustness hypothesis:

```bash
better-bench novelty \
  --benchmarks data/current \
  --models data/current \
  --observations data/current \
  --minimum-overlap 5 \
  --output-json novelty.json
```

The novelty command predicts each target result from empirically related, capability-similar benchmarks using **other models only**, then groups residuals by exposure evidence: guaranteed post-release, disclosed-cutoff unseen, sealed, rotating, short-lead likely unseen, possible exposure, or unknown.

See [`docs/first_data_diagnostic.md`](docs/first_data_diagnostic.md), [`docs/second_density_diagnostic.md`](docs/second_density_diagnostic.md), and [`docs/third_density_diagnostic.md`](docs/third_density_diagnostic.md) for empirical sanity checks.

## Benchmark evidence quality

Better Bench currently scores benchmark utility from measurement quality, adoption breadth, frontier discrimination, contamination/integrity, and empirical independence. The components are combined geometrically so a severe weakness cannot be hidden by popularity.

Evidence tiers are:

- **Core** — mature, discriminative and sufficiently high-integrity to influence the eventual headline capability index.
- **High-value emerging** — strong measurement evidence but not broad enough yet to dominate a general ranking.
- **Supporting** — useful domain evidence with material limitations, including highly adopted public benchmarks whose integrity is too weak for Core.
- **Diagnostic only** — retained for transparency/history but given little or no final-index influence.

Core status now has an explicit integrity floor. A widely run benchmark can therefore have a high raw importance score while remaining Supporting if contamination/exposure risk is too high. This prevents adoption from becoming a proxy for quality.

Benchmark categories that share a `family_id` split one family evidence budget. This prevents suites such as LiveBench—or multiple OSWorld protocols—from receiving independent full votes merely because multiple category or protocol scores are published.

The benchmark-quality layer is currently kept **separate from the unweighted sparse-factor diagnostic**. This is deliberate: the factor model should be able to disagree with our quality priors. The eventual hierarchical leaderboard model will introduce benchmark-quality and family weights explicitly, with sensitivity analysis rather than silently baking them into the exploratory factor fit.

## Data model

A benchmark definition specifies its fixed score scale, publication/exposure metadata, protocol quality, reliability, contamination-resistance metadata, family identity and capability loadings. A result row identifies the exact model plus provenance and optional system configuration:

```text
model_id, benchmark_id, score, evaluated_at, source_grade,
harness, reasoning_effort, token_budget, source_url,
model_revision, model_revision_at
```

Source grades are A (independent/benchmark-owner standardized), B (official controlled submission), C (documented vendor self-report), D (secondary aggregation), and E (unclear/unverified).

## Roadmap

1. **Public benchmark registry** — canonical benchmark/version metadata and source adapters.
2. **Frontier observation dataset** — model × benchmark × harness × effort with provenance.
3. **Empirical benchmark map** — correlations, clustering, redundancy and frontier discrimination.
4. **Benchmark Quality Index** — maturity/adoption, reliability, discrimination, contamination resistance, redundancy and family budgets.
5. **Novelty robustness model** — test exposure-correlated residual degradation on protected/post-training benchmarks.
6. **Bayesian latent capability model** — uncertainty-aware sparse-matrix inference with general + domain factors.
7. **IRT where item data exists** — item difficulty/discrimination instead of aggregate-only scoring.
8. **Capability Atlas UI** — profiles, confidence bands, benchmark diagnostics and pairwise comparison.

## Status

V0 now includes the schema, benchmark-importance model, family weighting, contamination heuristic, redundancy penalty, sparse capability profiles, a **414-observation / 40-model real-data corpus**, pairwise diagnostics, model-level residual analysis, and an explicit novelty-robustness detector.

The current factor diagnostic retains **24 models × 28 benchmarks with 328 measured cells (48.8% density)**. Rank 1 explains about **40.6%** of observed standardized variance and three factors explain about **73.8%**. The expansion therefore increased the dense analytical core slightly even though the full-registry raw density fell mechanically when twelve sparsely measured models were added.

The benchmark-quality model now classifies **MCP Atlas, SWE Atlas Codebase QnA, DeepSWE, Terminal-Bench 4 and Terminal-Bench Science** among the current Core evidence. The official release-pinned OSWorld 2.0 protocol remains high-value emerging. Highly adopted but public/low-integrity evidence such as MultiNRC remains Supporting rather than being promoted solely by popularity.

This is sufficient for useful structural diagnostics, but not yet a defensible final public scalar ranking across all 40 models. The next data milestone is to fill independent high-value cells for the sixteen models outside the current dense factor cohort, then fit a **general-factor + domain-residual model with explicit benchmark-quality, family, protocol and exposure effects**.
