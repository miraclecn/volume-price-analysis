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


def write_portfolio_diagnostics_report(
    diagnostics: pd.DataFrame,
    report_dir: Path | str,
    prefix: str = "portfolio_diagnostics",
) -> dict[str, Path]:
    path = Path(report_dir)
    path.mkdir(parents=True, exist_ok=True)
    metrics_path = path / f"{prefix}_metrics.csv"
    distribution_path = path / f"{prefix}_selected_count_distribution.csv"
    metrics = portfolio_diagnostics_report_metrics(diagnostics)
    pd.DataFrame(
        [{"metric_name": key, "metric_value": value} for key, value in metrics.items()]
    ).to_csv(metrics_path, index=False)
    selected_count_distribution(diagnostics).to_csv(distribution_path, index=False)
    return {
        "metrics": metrics_path,
        "selected_count_distribution": distribution_path,
    }


def portfolio_diagnostics_report_metrics(diagnostics: pd.DataFrame) -> dict[str, float]:
    if diagnostics.empty:
        return {
            "avg_raw_candidate_count": 0.0,
            "avg_hard_filter_pass_count": 0.0,
            "avg_core_pool_size": 0.0,
            "avg_candidate_pool_size": 0.0,
            "avg_selected_from_core": 0.0,
            "avg_selected_from_candidate": 0.0,
            "low_adv_rejected_count": 0.0,
            "cannot_buy_rejected_count": 0.0,
            "st_rejected_count": 0.0,
            "max_new_entries_blocked_count": 0.0,
            "sell_blocked_count": 0.0,
            "hold_due_to_min_days_count": 0.0,
            "exit_due_to_score_count": 0.0,
            "exit_due_to_risk_count": 0.0,
            "exit_due_to_time_count": 0.0,
            "empty_day_ratio": 0.0,
            "avg_selected_count": 0.0,
        }
    return {
        "avg_raw_candidate_count": _mean(diagnostics, "raw_candidate_count"),
        "avg_hard_filter_pass_count": _mean(diagnostics, "hard_filter_pass_count"),
        "avg_core_pool_size": _mean(diagnostics, "core_pool_size"),
        "avg_candidate_pool_size": _mean(diagnostics, "candidate_pool_size"),
        "avg_selected_from_core": _mean(diagnostics, "selected_from_core"),
        "avg_selected_from_candidate": _mean(diagnostics, "selected_from_candidate"),
        "low_adv_rejected_count": _sum(diagnostics, "low_adv_rejected_count"),
        "cannot_buy_rejected_count": _sum(diagnostics, "cannot_buy_rejected_count"),
        "st_rejected_count": _sum(diagnostics, "st_rejected_count"),
        "max_new_entries_blocked_count": _sum(diagnostics, "max_new_entries_blocked_count"),
        "sell_blocked_count": _sum(diagnostics, "sell_blocked_count"),
        "hold_due_to_min_days_count": _sum(diagnostics, "hold_due_to_min_days_count"),
        "exit_due_to_score_count": _sum(diagnostics, "exit_due_to_score_count"),
        "exit_due_to_risk_count": _sum(diagnostics, "exit_due_to_risk_count"),
        "exit_due_to_time_count": _sum(diagnostics, "exit_due_to_time_count"),
        "empty_day_ratio": float((pd.to_numeric(diagnostics.get("final_selected_count", pd.Series(dtype=float)), errors="coerce").fillna(0) == 0).mean()),
        "avg_selected_count": _mean(diagnostics, "final_selected_count"),
    }


def selected_count_distribution(diagnostics: pd.DataFrame) -> pd.DataFrame:
    if diagnostics.empty or "final_selected_count" not in diagnostics:
        return pd.DataFrame(columns=["final_selected_count", "day_count"])
    counts = (
        pd.to_numeric(diagnostics["final_selected_count"], errors="coerce")
        .fillna(0)
        .astype(int)
        .value_counts()
        .rename_axis("final_selected_count")
        .reset_index(name="day_count")
        .sort_values("final_selected_count")
        .reset_index(drop=True)
    )
    return counts


def _mean(frame: pd.DataFrame, column: str) -> float:
    if column not in frame:
        return 0.0
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.mean()) if not values.empty else 0.0


def _sum(frame: pd.DataFrame, column: str) -> float:
    if column not in frame:
        return 0.0
    return float(pd.to_numeric(frame[column], errors="coerce").fillna(0.0).sum())
