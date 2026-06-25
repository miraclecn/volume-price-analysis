from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from scripts.run_profit_protect_daily_prediction import (
    alpha_risk_predictions_from_raw,
    append_feature_json_columns,
    write_predictions_to_live_db,
)
from ml_stock_selector.serving.live_sim import PROFIT_PROTECT_PORTFOLIO_ID, PROFIT_PROTECT_SCORE_VERSION


def test_append_feature_json_columns_adds_fundamental_values_and_preserves_existing_features() -> None:
    features = pd.Series(
        [
            json.dumps({"ret_5": 0.1, "fund_eps": 99.0}),
            json.dumps({"ret_5": -0.2}),
        ]
    )
    fundamentals = pd.DataFrame(
        [
            {"fund_eps": 1.2, "fund_roe": 0.15},
            {"fund_eps": None, "fund_roe": 0.07},
        ]
    )

    out = append_feature_json_columns(features, fundamentals)

    first = json.loads(out.iloc[0])
    second = json.loads(out.iloc[1])
    assert first["ret_5"] == pytest.approx(0.1)
    assert first["fund_eps"] == pytest.approx(1.2)
    assert first["fund_roe"] == pytest.approx(0.15)
    assert second["ret_5"] == pytest.approx(-0.2)
    assert second["fund_eps"] == pytest.approx(0.0)
    assert second["fund_roe"] == pytest.approx(0.07)


def test_alpha_risk_predictions_from_raw_uses_daily_percent_rank_and_live_identity() -> None:
    raw = pd.DataFrame(
        [
            {
                "trade_date": "2026-06-16",
                "code": "b",
                "run_id": "run",
                "fold_id": "wf_2026",
                "score_version": "score",
                "feature_set_id": "features",
                "horizon_d": 5,
                "absolute_model_id": "alpha",
                "active_model_id": None,
                "risk_model_id": "risk",
                "absolute_score": 0.2,
                "active_score": None,
                "risk_prob": 0.8,
                "generated_at": "now",
            },
            {
                "trade_date": "2026-06-16",
                "code": "a",
                "run_id": "run",
                "fold_id": "wf_2026",
                "score_version": "score",
                "feature_set_id": "features",
                "horizon_d": 5,
                "absolute_model_id": "alpha",
                "active_model_id": None,
                "risk_model_id": "risk",
                "absolute_score": 0.1,
                "active_score": None,
                "risk_prob": 0.4,
                "generated_at": "now",
            },
        ]
    )

    out = alpha_risk_predictions_from_raw(raw).set_index("code")

    assert out.loc["a", "absolute_rank_pct"] == pytest.approx(0.0)
    assert out.loc["b", "absolute_rank_pct"] == pytest.approx(1.0)
    assert out.loc["a", "risk_rank_pct"] == pytest.approx(0.0)
    assert out.loc["b", "risk_rank_pct"] == pytest.approx(1.0)
    assert out.loc["a", "model_id"] == "alpha_risk:alpha:risk"
    assert out.loc["a", "trade_score_v2"] == pytest.approx(out.loc["a", "absolute_rank_pct"])
    assert out.loc["a", "run_id"] == "run"
    assert out.loc["a", "fold_id"] == "wf_2026"
    assert out.loc["a", "score_version"] == "score"


def test_write_predictions_to_live_db_records_live_owned_recent_predictions(tmp_path) -> None:
    predictions = pd.DataFrame(
        [
            {
                "trade_date": "2026-06-16",
                "code": "000001.SZ",
                "model_id": "alpha_risk:alpha:risk",
                "horizon_d": 5,
                "absolute_score": 0.1,
                "absolute_rank_pct": 0.2,
                "active_score": None,
                "active_rank_pct": 0.2,
                "risk_prob": 0.3,
                "risk_rank_pct": 0.4,
                "trade_score_v2": 0.5,
                "adv20_amount": 20_000_000.0,
                "generated_at": "now",
            }
        ]
    )

    live_db = tmp_path / "live.duckdb"

    written = write_predictions_to_live_db(predictions, live_db)
    con = duckdb.connect(str(live_db))
    active = con.execute("select * from live_model_bundle where is_active = true").fetchdf().iloc[0].to_dict()
    con.close()
    artifact_root = live_db.parent / "artifacts" / PROFIT_PROTECT_PORTFOLIO_ID / PROFIT_PROTECT_SCORE_VERSION

    assert written == 1
    assert Path(str(active["alpha_artifact_uri"])).is_relative_to(artifact_root)
    assert Path(str(active["risk_artifact_uri"])).is_relative_to(artifact_root)
    assert Path(str(active["feature_schema_uri"])).is_relative_to(artifact_root)
