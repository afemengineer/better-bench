# Current public-data seed

Snapshot assembled on **2026-09-04** to test whether Better Bench has enough real overlap to support a multidimensional model.

This is intentionally a **research seed**, not a production leaderboard. Scores are stored only when the source and protocol are explicit enough to distinguish materially different evaluation setups.

## Contents

- `models.yaml` — 28 tracked frontier/recent models.
- `benchmarks.yaml` — 21 protocol-specific benchmark definitions and hand-authored capability loadings.
- `observations.csv` — 136 source-tagged observations.

The raw matrix is sparse: 136 / (28 × 21) = **23.1% density**. Missing cells are missing evidence, never zeroes.

## Primary sources

| Dataset slice | Source | Grade used | Notes |
|---|---|---:|---|
| DeepSWE v1.1 | https://deepswe.datacurve.ai/ | A | Current official board, 113 tasks, mini-swe-agent for consistent runs. |
| Terminal-Bench 4.0 | https://snorkel.ai/leaderboard/terminal-bench-4-0/ | B | Mirror of official Harbor submissions; harness is retained per row. |
| Terminal-Bench-Science 0.1 | https://www.terminal-bench-science.ai/ | A | Original public benchmark/leaderboard. |
| HealthBench Professional independent run | https://healthbenchprofessional.com/ | A | Same task set/grader protocol across models; raw API, no tools. |
| AutomationBench public set | https://github.com/zapier/AutomationBench | A | Benchmark-author public baseline table. |
| CAISI / NIST suite | https://www.nist.gov/news-events/news/2026/05/caisi-evaluation-deepseek-v4-pro | A | Controlled evaluation across cyber, SWE, science, abstract reasoning and math. |
| GPT-6 Astra launch comparison | https://openai.com/index/gpt-6-astra/ | C | Vendor self-report; used for new models/benchmarks not yet represented in stronger independent sources. |

## Protocol separation rules

A benchmark name is not enough to establish comparability.

Examples:

- CAISI's SWE-Bench Verified is stored as `nist-swe-bench-verified` because CAISI explicitly warns its prompt, scaffolding and token budget differ from other leaderboards.
- CAISI ARC-AGI-2 Semi-Private is separate from the ARC-AGI-2 comparison reported in the GPT-6 Astra launch because the aggregation/protocol differs.
- HealthBench Professional independent raw-API runs are separate from OpenAI launch-report HealthBench scores.
- AutomationBench public-set scores are separate from held-out/private vendor comparison scores.

Terminal-Bench 4.0 and DeepSWE observations retain the **agent harness and reasoning effort** because those benchmarks measure model-plus-agent systems to a nontrivial degree.

## Known metadata debt

Some benchmark publication dates are currently month-level placeholders normalized to the first day of the month. Those rows say so in `notes`. They must be replaced with canonical dates before benchmark-age weighting is treated as production quality.

Likewise, the current taxonomy has no generic structured-tool-use factor. AutomationBench is provisionally loaded partly onto `web_agency`; this is deliberately marked as questionable rather than hidden.
