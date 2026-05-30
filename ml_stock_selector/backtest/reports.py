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
