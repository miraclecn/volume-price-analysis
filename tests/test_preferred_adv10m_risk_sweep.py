from __future__ import annotations

import pandas as pd

from scripts.run_preferred_adv10m_risk_sweep import (
    RiskVariant,
    apply_preferred_adv015_score,
    build_portfolio_id,
    continuous_nav_summary,
)


def test_preferred_adv015_score_uses_full_prediction_pool_adv_rank():
    candidates = pd.DataFrame(
        {
            "trade_date": ["2020-01-02", "2020-01-02", "2020-01-02"],
            "code": ["a", "b", "c"],
            "absolute_rank_pct": [0.90, 0.80, 0.70],
            "risk_rank_pct": [0.20, 0.30, 0.40],
            "adv20_amount": [100.0, 300.0, 200.0],
        }
    )

    scored = apply_preferred_adv015_score(candidates)

    assert scored["full_prediction_pool_adv_pct"].round(6).tolist() == [
        round(1 / 3, 6),
        1.0,
        round(2 / 3, 6),
    ]
    assert scored["trade_score_v2"].round(6).tolist() == [
        round(0.85 * 0.90 + 0.15 * (1 - 1 / 3), 6),
        round(0.85 * 0.80 + 0.15 * (1 - 1.0), 6),
        round(0.85 * 0.70 + 0.15 * (1 - 2 / 3), 6),
    ]
    assert scored["alpha_rank_pct"].tolist() == scored["absolute_rank_pct"].tolist()
    assert scored["active_rank_pct"].tolist() == scored["absolute_rank_pct"].tolist()


def test_build_portfolio_id_records_risk_variant():
    variant = RiskVariant(label="risk050", candidate_risk_max_rank_pct=0.50, core_risk_max_rank_pct=0.50)

    portfolio_id = build_portfolio_id("wf_2024", variant)

    assert portfolio_id == "wf_2024_absolute_risk_filter_score_adv_combo_top12_risk050_c050_core050"


def test_continuous_nav_summary_stitches_fold_returns_without_resetting_capital():
    nav = pd.DataFrame(
        {
            "sim_date": pd.to_datetime(["2020-01-02", "2020-12-31", "2021-01-04", "2021-12-31"]),
            "fold_id": ["wf_2020", "wf_2020", "wf_2021", "wf_2021"],
            "nav": [100.0, 130.0, 100.0, 120.0],
        }
    )

    stitched, summary = continuous_nav_summary(nav)

    assert stitched["continuous_nav"].round(6).tolist() == [100.0, 130.0, 130.0, 156.0]
    assert summary["total_return"] == 0.56
    assert summary["max_drawdown"] == 0.0
