# Provisional model mode

Better Bench separates **officially rankable** models from **provisional** models.

A model is officially rankable only when it survives the estimator's iterative overlap
filter and has evidence from at least five independent benchmark families. Several
protocol variants from one family do not count as independent evidence.

Models below that threshold remain visible rather than disappearing from the frontier.
For each provisional model Better Bench reports:

- a ridge-shrunk projected general-capability z-score;
- number of retained benchmark observations;
- number of independent benchmark families;
- effective family evidence and calibrated-family coverage;
- a confidence label and the explicit reason the model is not officially rankable.

## Critical separation

The provisional projection is calculated **after** the official estimator has learned
benchmark intercepts, discrimination loadings and evidence weights from the rankable
cohort. Provisional models do not refit or perturb those calibrations and therefore
cannot promote themselves into the official ranking through one unusually strong sparse
result.

The projection solves, in effect,

`score_mb ~= benchmark_intercept_b + benchmark_loading_b * provisional_g_m`

using only the sparse model's observations on benchmark protocols already calibrated by
the official cohort, with ridge shrinkage toward the frontier mean.

A provisional score is therefore useful for statements such as "current evidence points
to roughly frontier level, but coverage is insufficient." It is **not** assigned an
official rank, official confidence interval, or taxonomy top-five position.

Once the model gains enough independent-family evidence and survives the structural
overlap filter, it is removed from provisional mode and refit normally as part of the
official cohort.
