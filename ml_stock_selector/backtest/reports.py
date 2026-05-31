from __future__ import annotations

from pathlib import Path

import pandas as pd

from ml_stock_selector.backtest.metrics import summarize_unknown_industry_exposure


def write_metrics_report(metrics: dict[str, float], report_dir: Path | str, name: str = "metrics.csv") -> Path:
    path = Path(report_dir)
    path.mkdir(parents=True, exist_ok=True)
    out = path / name
    pd.DataFrame([{"metric_name": key, "metric_value": value} for key, value in metrics.items()]).to_csv(out, index=False)
    return out


def unknown_industry_report_metrics(daily_exposure: pd.DataFrame) -> dict[str, float]:
    return summarize_unknown_industry_exposure(daily_exposure)


def prediction_report_metrics(predictions: pd.DataFrame) -> dict[str, float]:
    if predictions.empty:
        return {
            "prediction_model_count": 0.0,
            "prediction_v2_three_model_rows": 0.0,
            "prediction_active_rank_pct_mean": 0.0,
            "prediction_risk_prob_mean": 0.0,
        }
    score_version = predictions.get("score_version", pd.Series(dtype=object))
    active_rank = pd.to_numeric(predictions.get("active_rank_pct", pd.Series(dtype=float)), errors="coerce")
    risk_prob = pd.to_numeric(predictions.get("risk_prob", pd.Series(dtype=float)), errors="coerce")
    return {
        "prediction_model_count": float(predictions.get("model_id", pd.Series(dtype=object)).nunique()),
        "prediction_v2_three_model_rows": float((score_version == "v2_three_model").sum()),
        "prediction_active_rank_pct_mean": float(active_rank.dropna().mean() if active_rank.notna().any() else 0.0),
        "prediction_risk_prob_mean": float(risk_prob.dropna().mean() if risk_prob.notna().any() else 0.0),
    }
