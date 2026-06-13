from __future__ import annotations

import pandas as pd

from ml_stock_selector.portfolio.allocator import allocate_weights


def test_allocate_weights_does_not_exceed_one_when_min_weight_conflicts_with_position_count():
    selected = pd.DataFrame(
        {
            "trade_date": ["2024-01-02"] * 12,
            "code": [f"s{idx:02d}" for idx in range(12)],
            "target_weight": [0.0] * 12,
        }
    )

    weighted = allocate_weights(selected, min_weight=0.10, max_weight=0.30, allow_cash=True)

    assert round(float(weighted["target_weight"].sum()), 10) == 1.0
    assert weighted["target_weight"].max() <= 0.30
