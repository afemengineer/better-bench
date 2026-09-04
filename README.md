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
- **Redundancy is penalized.** Highly correlated benchmarks should not receive ten votes for one skill.
- **Provenance matters.** Independent evaluations weigh more than secondary or unclear reports.
- **Model ≠ agent harness.** Harness, reasoning effort and token budget are retained in every observation.

The current estimator is intentionally transparent. It is scaffolding for a later hierarchical Bayesian / multidimensional IRT model, not a claim that weighted averages solve psychometrics. See [`docs/methodology.md`](docs/methodology.md) and [`docs/benchmaxxing_hypothesis.md`](docs/benchmaxxing_hypothesis.md).

## Capability taxonomy

V0 tracks 14 domains: fluid reasoning, quantitative reasoning, scientific reasoning, knowledge, language, software engineering, terminal agency, web agency, GUI/computer use, visual intelligence, spatial intelligence, long-context/state tracking, planning/agency, and social/pragmatic intelligence.

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

The synthetic example remains useful for tests. A real public-data seed now lives in [`data/current/`](data/current/) with 28 models, 21 protocol-specific benchmarks and 136 provenance-tagged observations.

Inspect empirical benchmark relationships:

```bash
better-bench diagnose \
  --benchmarks data/current/benchmarks.yaml \
  --observations data/current/observations.csv \
  --minimum-overlap 4 \
  --left deepswe-v1.1 \
  --right terminal-bench-4.0
```

Test the benchmaxxing / novelty-robustness hypothesis:

```bash
better-bench novelty \
  --benchmarks data/current/benchmarks.yaml \
  --models data/current/models.yaml \
  --observations data/current/observations.csv \
  --minimum-overlap 5 \
  --output-json novelty.json
```

The novelty command predicts each target result from empirically related, capability-similar benchmarks using **other models only**, then groups residuals by exposure evidence: guaranteed post-release, disclosed-cutoff unseen, sealed, rotating, short-lead likely unseen, possible exposure, or unknown.

See [`docs/first_data_diagnostic.md`](docs/first_data_diagnostic.md) for the first real-matrix sanity check.

## Data model

A benchmark definition specifies its fixed score scale, publication/exposure metadata, protocol quality, reliability and capability loadings. A result row identifies the exact model plus provenance and optional system configuration:

```text
model_id, benchmark_id, score, evaluated_at, source_grade,
harness, reasoning_effort, token_budget, source_url
```

Source grades are A (independent/benchmark-owner standardized), B (official controlled submission), C (documented vendor self-report), D (secondary aggregation), and E (unclear/unverified).

## Roadmap

1. **Public benchmark registry** — canonical benchmark/version metadata and source adapters.
2. **Frontier observation dataset** — model × benchmark × harness × effort with provenance.
3. **Empirical benchmark map** — correlations, clustering, redundancy and frontier discrimination.
4. **Novelty robustness model** — test exposure-correlated residual degradation on protected/post-training benchmarks.
5. **Bayesian latent capability model** — uncertainty-aware sparse-matrix inference with general + domain factors.
6. **IRT where item data exists** — item difficulty/discrimination instead of aggregate-only scoring.
7. **Capability Atlas UI** — profiles, confidence bands, benchmark diagnostics and pairwise comparison.
8. **Benchmark Quality Index** — freshness, contamination risk, reliability, discrimination and redundancy.

## Status

V0 now includes the schema, quality weighting, contamination heuristic, redundancy penalty, sparse capability profiles, a real public-data seed, pairwise diagnostics, model-level residual analysis, and a first explicit novelty-robustness detector.

The current matrix is still too sparse for a defensible public scalar ranking. The next data milestone is a **denser cross-domain core matrix** with enough old vs protected/new overlap to estimate a hierarchical model-specific exposure effect rather than relying on descriptive residuals.
