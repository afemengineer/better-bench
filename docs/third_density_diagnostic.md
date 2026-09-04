# Third density diagnostic — 40-model registry

Date: 2026-09-04

This pass deliberately optimized **overlap and benchmark utility**, not raw benchmark count. It expands the model registry from 28 to 40 while adding measurements that bridge software engineering, terminal/scientific agency, computer use, visual tool use, multilingual reasoning and multi-turn reliability.

## Corpus size

The current registry contains:

- **40 models**
- **41 protocol-specific benchmark definitions**
- **414 provenance-tagged observations**
- 1,640 possible model × benchmark cells
- **25.2% raw registry density**

The lower raw density versus the previous 28-model corpus is expected: twelve newly registered models were added only where credible public evidence was found instead of fabricating completeness from weak sources.

For structural factor analysis, Better Bench currently requires at least five benchmark observations per model and five models per benchmark. The retained matrix is:

- **24 models × 28 benchmarks**
- **328 / 672 observed cells**
- **48.8% retained density**

This is slightly denser than the previous retained matrix (24 × 27, 313 / 648, 48.3%). The distinction between the broad registry and the dense analytical cohort is therefore essential.

## What was added

### Model bridges

Twelve models were selected because they recur across serious benchmark families rather than because they are simply recent:

- Muse Spark 1.1
- Claude Opus 4.7
- Gemini 3.1 Pro
- GPT-5.4
- Gemini 3 Pro
- Claude Sonnet 4.6
- GPT-5.2
- Kimi K2.5
- Gemini 3.1 Flash-Lite
- Claude Sonnet 4.5
- GPT-5.1
- MiniMax M3

Registry membership does **not** imply ranking eligibility. Models remain outside the dense factor cohort until they accumulate enough independent benchmark evidence.

### Benchmark bridges

Five protocol-specific benchmarks were added:

1. **VisualToolBench** — visual perception plus structured tool use.
2. **MultiChallenge** — multi-turn instruction retention, inference memory and state consistency.
3. **OSWorld 2.0 v2026.06.24 binary** — release-pinned computer-use protocol with 500-step budget.
4. **MultiNRC** — native multilingual reasoning in French, Spanish and Chinese.
5. **SWE Atlas Codebase QnA** — codebase/runtime comprehension rather than patch generation.

OSWorld protocol variants now share the same `family_id`, so separate scoring protocols cannot create independent full votes for the same benchmark family.

## Benchmark-importance behavior

The expanded cohort materially changed benchmark maturity estimates.

### Terminal-Bench Science becomes Core

After adding seven non-duplicative benchmark-owner results, Terminal-Bench Science now has enough broad comparable evidence to classify as Core:

- importance: **0.755**
- adoption score: **0.874**
- frontier discrimination: **0.725**
- integrity: **0.950**

This is the intended behavior: a previously high-value emerging benchmark can graduate when actual adoption and comparable evidence increase.

### Popularity is no longer sufficient for Core

A new Core-tier gate requires benchmark integrity >= 0.60.

This matters immediately for MultiNRC:

- raw importance: **0.855**
- adoption: **1.000**
- discrimination: **0.940**
- measurement quality: **0.930**
- integrity: **0.450**
- final tier: **Supporting**

Without the integrity gate, a widely adopted public benchmark could become Core despite substantial contamination/exposure risk. The updated rule makes adoption useful evidence of benchmark maturity without turning popularity into quality.

### SWE Atlas Codebase QnA becomes high-value independent coding evidence

SWE Atlas Codebase QnA currently classifies Core:

- importance: **0.867**
- adoption: **0.962**
- discrimination: **0.978**
- integrity: **0.800**

Its exploratory general-factor loading is only about **+0.224**, despite very high benchmark quality. That is potentially valuable: it suggests codebase comprehension may carry substantial coding-specific information instead of acting as another near-duplicate proxy for general capability.

This should be tested with a proper hierarchical model before interpreting the loading causally.

## Factor structure after expansion

The unweighted missing-data factor diagnostic now gives:

- rank 1 cumulative explained observed variance: **40.6%**
- rank 2: **59.9%**
- rank 3: **73.8%**

The common factor therefore remains large but clearly incomplete.

The broad ordering of the existing dense cohort remains stable. GPT-6 Astra is the largest positive factor-1 score, followed by Claude Fable 5.1, Claude Fable 5, Claude Opus 5, GPT-5.5 and GPT-5.6 Sol. Gemini 3.8 Flash remains near the middle/lower-middle of this exploratory 24-model cohort rather than at the bottom after the prior HLE/OSWorld correction.

The 12 newly registered models are not yet permitted to distort this ordering merely because one or two strong benchmark results exist.

## Novelty robustness

The expanded corpus produces:

- **315 usable leave-target-out residuals**
- 66 strong-unseen
- 36 protected
- 75 suggestive-unseen
- 69 possible-exposure
- descriptive global novelty gap: **-1.02**
- naive 95% interval: **-2.99 to +0.95**

The interval still spans zero. There is still no strong evidence for a frontier-wide generic "benchmaxxing penalty."

A new large anomaly appears for Gemini 3.1 Pro on SWE Atlas Codebase QnA (13.5 observed versus about 42 predicted), but it currently has only one usable comparator and should **not** be treated as robust evidence. Gemini 3.8 Flash on Terminal-Bench 4 remains a better-supported model-specific anomaly because it is calibrated from several comparators.

## Next acquisition target

The bottleneck is no longer model registry size. It is **evidence depth for the sixteen models outside the dense factor cohort**.

The next data sweep should prioritize benchmark cells that simultaneously:

1. move one or more of those models toward the five-benchmark threshold;
2. come from Core or high-value-emerging benchmark families;
3. overlap with the existing 24-model calibration cohort;
4. add a capability direction not already overrepresented for that model;
5. preserve protocol/harness identity rather than merging superficially similar results.

Likely high-leverage sources include additional SWE Atlas domains, protected/private software-engineering evaluations, release-pinned computer-use results, and independent multilingual/visual reasoning benchmarks. The objective is to turn the 40-model registry into a 30+ model dense calibration cohort before fitting the final hierarchical general + domain model.
