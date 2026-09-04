# Benchmark evidence importance

Better Bench does not treat every benchmark as one equal vote. Benchmark popularity matters because broad adoption improves overlap, comparability, and the ability to estimate benchmark behavior, but popularity alone is not evidence of measurement quality.

## Two separate concepts

1. **Evidence importance**: how much a benchmark deserves to influence a final capability estimate.
2. **Maturity / adoption**: how broadly the benchmark has been run across models and organizations.

A new sealed benchmark can therefore be high-value but immature, while an old famous benchmark can be mature but nearly useless at the frontier because it is saturated.

## Importance components

The V0 Benchmark Quality Index uses five components:

- **Measurement quality (30%)** — protocol quality and reliability.
- **Adoption (20%)** — breadth of model configurations and organizations represented on the leaderboard. Model counts are log-scaled; adoption saturates rather than growing without bound.
- **Frontier discrimination (25%)** — current score spread plus remaining headroom from ceiling/floor saturation.
- **Integrity / contamination resistance (15%)** — sealed/rotating status, explicit benchmark metadata, and age when no stronger evidence exists.
- **Independence (10%)** — redundancy penalty derived from cross-model benchmark correlations.

These components are combined with a weighted **geometric mean**, not an arithmetic mean. A severe weakness therefore cannot be completely compensated by being popular. In particular, a benchmark with 100 leaderboard entries but almost no frontier score variance can still be diagnostic-only.

The exact V0 weights are hypotheses and should be sensitivity-tested later. They are not intended as permanent constants.

## Evidence tiers

### Core
Mature, sufficiently broad, discriminative, reliable benchmark evidence. Eligible to contribute materially to the final general-capability estimate.

### High-value emerging
Strong protocol/integrity/discrimination but not enough adoption yet. Especially valuable for novelty robustness and for discovering missing dimensions, but should not dominate a general leaderboard until the model cohort broadens.

### Supporting
Useful additional evidence, often because it covers a needed domain, but limited by contamination risk, redundancy, saturation, protocol provenance, or adoption.

### Diagnostic only
Kept for transparency, historical comparison, or specific capability analysis, but should have little or no influence on the final general index. A famous benchmark can land here if frontier saturation has destroyed its current measurement value.

## Benchmark families

A benchmark suite must not gain extra influence simply by publishing many category columns or overlapping subsets. Each benchmark definition can therefore declare a `family_id`.

V0 gives each family one evidence budget equal to the strongest member's raw importance. Sub-benchmarks split that budget in proportion to their individual importance. This preserves category-level information while preventing, for example, seven LiveBench categories from counting as seven independent benchmark ecosystems.

Family adjustment is intended for final evidence weighting and acquisition prioritization. It does **not** alter the current unweighted sparse factor diagnostic; keeping those analyses separate lets us detect when the learned latent structure disagrees with our benchmark-quality priors.

## Adoption snapshots are time-dependent

Popularity is stored as dated adoption snapshots rather than permanent benchmark metadata. A newly launched benchmark can move from emerging to core as more independent model organizations evaluate it. Conversely, a formerly important benchmark can decline as it saturates or becomes redundant.

## What the quality layer is for

The Benchmark Quality Index should be used for:

- deciding which benchmark results are worth collecting next;
- gating benchmarks into the eventual public index;
- weighting evidence/confidence in the final hierarchical model;
- exposing why a benchmark is included or down-weighted;
- preventing benchmark farms and highly subdivided suites from dominating by count.

It should **not** be used to force the latent capability model to agree with a preferred leaderboard ordering.
