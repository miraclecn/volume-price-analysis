from __future__ import annotations

import pandas as pd

from scripts.build_ml_labels import _label_load_end_date, _trim_labels_to_requested_range


def test_label_load_end_date_extends_requested_end_for_future_window():
    assert _label_load_end_date("2024-01-31", [1, 5, 10]) > "2024-01-31"


def test_trim_labels_keeps_only_requested_trade_dates():
    labels = pd.DataFrame(
        [
            {"trade_date": "2024-01-30", "code": "a"},
            {"trade_date": "2024-01-31", "code": "a"},
            {"trade_date": "2024-02-01", "code": "a"},
        ]
    )

    trimmed = _trim_labels_to_requested_range(labels, "2024-01-31", "2024-01-31")

    assert trimmed["trade_date"].tolist() == ["2024-01-31"]
