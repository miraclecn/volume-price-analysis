from __future__ import annotations

import pandas as pd


def cross_sectional_percentile(frame: pd.DataFrame, score_column: str, date_column: str = "trade_date") -> pd.Series:
    return frame.groupby(date_column)[score_column].rank(pct=True).fillna(0.5)

