from types import SimpleNamespace

from better_bench.pairwise_ranking import PairwiseRankingConfig, fit_pairwise_ranking


def _row(model, benchmark, family, score, weight=1.0):
    return SimpleNamespace(
        model_id=model,
        benchmark_id=benchmark,
        family_id=family,
        score_points=float(score),
        weight=float(weight),
    )


def _state(general, loadings):
    return SimpleNamespace(general=general, loading=loadings)


def test_shared_evidence_can_correct_portfolio_biased_prior():
    model_ids = ["strong", "sparse", "anchor"]
    rows = [
        _row("strong", "b1", "f1", 70),
        _row("sparse", "b1", "f1", 60),
        _row("anchor", "b1", "f1", 45),
        _row("strong", "b2", "f2", 75),
        _row("sparse", "b2", "f2", 62),
        _row("anchor", "b2", "f2", 48),
        # Only the better-covered model and anchor ran this difficult benchmark.
        _row("strong", "hard", "f3", 32),
        _row("anchor", "hard", "f3", 18),
    ]
    state = _state(
        # Simulate the exact failure mode: the portfolio-based prior ranks the sparse
        # model above the stronger model because it never saw the hard benchmark.
        {"strong": 0.2, "sparse": 0.8, "anchor": -0.7},
        {"b1": 1.0, "b2": 1.0, "hard": 1.0},
    )

    result = fit_pairwise_ranking(
        rows,
        model_ids,
        state,
        score_unit_points=10.0,
        config=PairwiseRankingConfig(minimum_shared_families=2, ridge=1.0),
    )

    assert result.prior_scores["sparse"] > result.prior_scores["strong"]
    assert result.scores["strong"] > result.scores["sparse"]


def test_missing_cells_reduce_graph_information_instead_of_creating_votes():
    model_ids = ["a", "b", "c", "isolated"]
    rows = [
        _row("a", "b1", "f1", 70),
        _row("b", "b1", "f1", 60),
        _row("c", "b1", "f1", 50),
        _row("a", "b2", "f2", 75),
        _row("b", "b2", "f2", 65),
        _row("c", "b2", "f2", 55),
    ]
    state = _state(
        {"a": 0.8, "b": 0.2, "c": -0.4, "isolated": 0.0},
        {"b1": 1.0, "b2": 1.0},
    )

    result = fit_pairwise_ranking(
        rows,
        model_ids,
        state,
        score_unit_points=10.0,
        config=PairwiseRankingConfig(minimum_shared_families=2, ridge=1.0),
    )

    assert result.graph_information["isolated"] == 0.0
    assert result.graph_information["a"] > 0.0
    assert result.graph_standard_error["isolated"] > result.graph_standard_error["a"]


def test_neutral_prior_does_not_reimport_portfolio_score_for_isolated_model():
    model_ids = ["a", "b", "isolated"]
    rows = [
        _row("a", "b1", "f1", 70),
        _row("b", "b1", "f1", 50),
        _row("a", "b2", "f2", 75),
        _row("b", "b2", "f2", 55),
    ]
    state = _state(
        # The isolated model has an implausibly high old portfolio score. With a neutral
        # prior it must not inherit that score through graph regularization.
        {"a": 0.3, "b": -0.4, "isolated": 3.0},
        {"b1": 1.0, "b2": 1.0},
    )

    neutral = fit_pairwise_ranking(
        rows,
        model_ids,
        state,
        score_unit_points=10.0,
        config=PairwiseRankingConfig(
            minimum_shared_families=2,
            ridge=1.0,
            prior_mix=0.0,
        ),
    )
    portfolio_prior = fit_pairwise_ranking(
        rows,
        model_ids,
        state,
        score_unit_points=10.0,
        config=PairwiseRankingConfig(
            minimum_shared_families=2,
            ridge=1.0,
            prior_mix=1.0,
        ),
    )

    assert neutral.graph_information["isolated"] == 0.0
    assert abs(neutral.scores["isolated"]) < 1e-9
    assert portfolio_prior.scores["isolated"] > neutral.scores["isolated"]


def test_protocol_variants_inside_one_family_do_not_satisfy_independence_threshold():
    model_ids = ["a", "b"]
    rows = [
        _row("a", "variant1", "same_family", 70),
        _row("b", "variant1", "same_family", 60),
        _row("a", "variant2", "same_family", 72),
        _row("b", "variant2", "same_family", 61),
    ]
    state = _state(
        {"a": 0.5, "b": -0.5},
        {"variant1": 1.0, "variant2": 1.0},
    )

    result = fit_pairwise_ranking(
        rows,
        model_ids,
        state,
        score_unit_points=10.0,
        config=PairwiseRankingConfig(minimum_shared_families=2, ridge=1.0),
    )

    assert result.edges == []
    assert result.scores["a"] > result.scores["b"]
