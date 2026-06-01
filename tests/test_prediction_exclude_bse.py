from __future__ import annotations

import pandas as pd

from ml_stock_selector.universe import apply_universe_filter


def test_prediction_side_excludes_bse_candidates():
    frame = pd.DataFrame(
        {
            "trade_date": ["2024-01-02", "2024-01-02", "2024-01-02"],
            "code": ["430001.BJ", "000001.SZ", "000002.SZ"],
            "feature_set_id": ["x", "x", "x"],
        }
    )
    out = apply_universe_filter(frame, exclude_bse=True)
    assert set(out["code"]) == {"000001.SZ", "000002.SZ"}

