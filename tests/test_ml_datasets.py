from __future__ import annotations

import pandas as pd

from ml_stock_selector.datasets import DateRange, build_lgbm_group, make_walk_forward_split


def test_walk_forward_split_applies_embargo_and_groups_by_date():
    frame = pd.DataFrame(
        {
            "trade_date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
            "code": ["a", "a", "a", "a"],
        }
    )

    split = make_walk_forward_split(
        frame,
        DateRange("2024-01-01", "2024-01-03"),
        DateRange("2024-01-04", "2024-01-04"),
        DateRange("2024-01-04", "2024-01-04"),
        embargo_days=1,
    )

    assert split.train["trade_date"].tolist() == ["2024-01-01", "2024-01-02"]
    assert build_lgbm_group(pd.DataFrame({"trade_date": ["d1", "d1", "d2"], "code": ["a", "b", "a"]})) == [2, 1]

