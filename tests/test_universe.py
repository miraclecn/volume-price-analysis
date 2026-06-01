from __future__ import annotations

import pandas as pd

from ml_stock_selector.universe import apply_universe_filter, detect_is_bse


def test_detect_is_bse_from_suffix():
    assert detect_is_bse("430001.BJ")
    assert not detect_is_bse("000001.SZ")


def test_apply_universe_filter_excludes_bse():
    frame = pd.DataFrame(
        {
            "trade_date": ["2024-01-02", "2024-01-02"],
            "code": ["430001.BJ", "000001.SZ"],
        }
    )
    filtered = apply_universe_filter(frame, exclude_bse=True)
    assert filtered["code"].tolist() == ["000001.SZ"]
