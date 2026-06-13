from __future__ import annotations

import pandas as pd

from scripts.run_ml_backtest import (
    SCORE_VERSION_ABSOLUTE_RISK_SORT,
    SCORE_VERSION_ABSOLUTE_RISK_FILTER,
    SCORE_VERSION_ABSOLUTE_ONLY,
    SCORE_VERSION_THREE_MODEL,
    _apply_score_mode,
    _apply_constraint_overrides,
    _portfolio_id_for_mode,
    _score_version_for_mode,
    build_arg_parser,
)
from ml_stock_selector.portfolio.constraints import PortfolioConstraints


def test_backtest_cli_defaults_to_three_model_score_mode():
    args = build_arg_parser().parse_args(["--config", "config/ml_walkforward.toml", "--fold-id", "wf_2020"])

    assert args.score_mode == "three_model"
    assert args.portfolio_suffix is None
    assert args.candidate_risk_max_rank_pct is None
    assert args.core_risk_max_rank_pct is None
    assert args.target_positions is None
    assert args.selection_buckets is None
    assert args.selection_per_bucket is None
    assert _score_version_for_mode(args.score_mode) == SCORE_VERSION_THREE_MODEL
    assert _portfolio_id_for_mode(args.fold_id, args.score_mode) == "wf_2020"


def test_absolute_only_score_mode_uses_absolute_rank_and_neutralizes_other_models():
    scored = pd.DataFrame(
        {
            "code": ["a", "b"],
            "absolute_rank_pct": [0.76, 0.82],
            "active_rank_pct": [0.10, 0.20],
            "risk_rank_pct": [0.90, 0.95],
            "risk_prob": [0.80, 0.70],
            "trade_score_v2": [0.20, 0.30],
            "score_version": [SCORE_VERSION_THREE_MODEL, SCORE_VERSION_THREE_MODEL],
        }
    )

    adjusted = _apply_score_mode(scored, "absolute_only")

    assert adjusted["trade_score_v2"].tolist() == [0.76, 0.82]
    assert adjusted["active_rank_pct"].tolist() == [0.76, 0.82]
    assert adjusted["risk_rank_pct"].tolist() == [0.0, 0.0]
    assert adjusted["risk_prob"].tolist() == [0.0, 0.0]
    assert adjusted["score_version"].eq(SCORE_VERSION_ABSOLUTE_ONLY).all()
    assert _portfolio_id_for_mode("wf_2020", "absolute_only") == "wf_2020_absolute_only"


def test_absolute_risk_filter_score_mode_uses_absolute_rank_and_keeps_risk_model():
    scored = pd.DataFrame(
        {
            "code": ["a", "b"],
            "absolute_rank_pct": [0.76, 0.82],
            "active_rank_pct": [0.10, 0.20],
            "risk_rank_pct": [0.60, 0.95],
            "risk_prob": [0.30, 0.80],
            "trade_score_v2": [0.20, 0.30],
            "score_version": [SCORE_VERSION_THREE_MODEL, SCORE_VERSION_THREE_MODEL],
        }
    )

    adjusted = _apply_score_mode(scored, "absolute_risk_filter")

    assert adjusted["trade_score_v2"].tolist() == [0.76, 0.82]
    assert adjusted["active_rank_pct"].tolist() == [0.76, 0.82]
    assert adjusted["risk_rank_pct"].tolist() == [0.60, 0.95]
    assert adjusted["risk_prob"].tolist() == [0.30, 0.80]
    assert adjusted["score_version"].eq(SCORE_VERSION_ABSOLUTE_RISK_FILTER).all()
    assert _portfolio_id_for_mode("wf_2020", "absolute_risk_filter") == "wf_2020_absolute_risk_filter"


def test_absolute_risk_sort_score_mode_scores_with_absolute_and_risk_without_filtering_risk():
    scored = pd.DataFrame(
        {
            "code": ["a", "b"],
            "absolute_rank_pct": [0.90, 0.80],
            "active_rank_pct": [0.10, 0.20],
            "risk_rank_pct": [0.10, 0.90],
            "risk_prob": [0.30, 0.80],
            "trade_score_v2": [0.20, 0.30],
            "score_version": [SCORE_VERSION_THREE_MODEL, SCORE_VERSION_THREE_MODEL],
        }
    )

    adjusted = _apply_score_mode(scored, "absolute_risk_sort")

    assert adjusted["trade_score_v2"].tolist() == [0.75, 0.545]
    assert adjusted["active_rank_pct"].tolist() == [0.90, 0.80]
    assert adjusted["risk_rank_pct"].tolist() == [0.10, 0.90]
    assert adjusted["risk_prob"].tolist() == [0.30, 0.80]
    assert adjusted["score_version"].eq(SCORE_VERSION_ABSOLUTE_RISK_SORT).all()
    assert _portfolio_id_for_mode("wf_2022", "absolute_risk_sort") == "wf_2022_absolute_risk_sort"


def test_three_model_score_mode_leaves_scores_unchanged():
    scored = pd.DataFrame({"absolute_rank_pct": [0.9], "trade_score_v2": [0.7]})

    adjusted = _apply_score_mode(scored, "three_model")

    pd.testing.assert_frame_equal(adjusted, scored)


def test_backtest_cli_accepts_risk_threshold_overrides_and_portfolio_suffix():
    args = build_arg_parser().parse_args(
        [
            "--fold-id",
            "wf_2021",
            "--score-mode",
            "absolute_risk_filter",
            "--candidate-risk-max-rank-pct",
            "0.55",
            "--core-risk-max-rank-pct",
            "0.45",
            "--portfolio-suffix",
            "risk055_045",
            "--target-positions",
            "15",
            "--selection-buckets",
            "4",
            "--selection-per-bucket",
            "3",
        ]
    )

    assert args.candidate_risk_max_rank_pct == 0.55
    assert args.core_risk_max_rank_pct == 0.45
    assert args.target_positions == 15
    assert args.selection_buckets == 4
    assert args.selection_per_bucket == 3
    assert _portfolio_id_for_mode(args.fold_id, args.score_mode, args.portfolio_suffix) == (
        "wf_2021_absolute_risk_filter_risk055_045"
    )


def test_target_position_override_expands_matching_position_caps():
    args = build_arg_parser().parse_args(
        ["--target-positions", "15", "--selection-buckets", "4", "--selection-per-bucket", "3"]
    )
    constraints = PortfolioConstraints(target_positions=12, hard_max_positions=15, max_initial_entries=12)

    adjusted = _apply_constraint_overrides(constraints, args)

    assert adjusted.target_positions == 15
    assert adjusted.hard_max_positions == 15
    assert adjusted.max_initial_entries == 15
    assert adjusted.selection_bucket_count == 4
    assert adjusted.selection_per_bucket == 3


def test_absolute_risk_sort_constraint_defaults_use_absolute_gate_not_risk_gate():
    args = build_arg_parser().parse_args(["--score-mode", "absolute_risk_sort"])
    constraints = PortfolioConstraints(
        candidate_min_trade_score=0.75,
        core_min_trade_score=0.75,
        candidate_absolute_min_rank_pct=0.70,
        core_absolute_min_rank_pct=0.75,
        candidate_risk_max_rank_pct=0.55,
        core_risk_max_rank_pct=0.55,
    )

    adjusted = _apply_constraint_overrides(constraints, args)

    assert adjusted.candidate_min_trade_score == 0.451
    assert adjusted.core_min_trade_score == 0.451
    assert adjusted.candidate_absolute_min_rank_pct == 0.75
    assert adjusted.core_absolute_min_rank_pct == 0.75
    assert adjusted.candidate_risk_max_rank_pct == 1.0
    assert adjusted.core_risk_max_rank_pct == 1.0
