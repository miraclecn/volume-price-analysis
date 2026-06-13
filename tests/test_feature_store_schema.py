from __future__ import annotations

import json

from ml_stock_selector.feature_store import DENIED_FEATURE_COLUMNS, parse_numeric_features


def test_feature_store_schema_parser_keeps_only_numeric_training_features():
    parsed = parse_numeric_features(
        json.dumps(
            {
                "ret_1d": 0.1,
                "range_pct": 0.02,
                "flag": True,
                "raw_label": "NORMAL_UP",
                "industry_name": "Industry",
                "can_buy_next_open": True,
            }
        )
    )

    assert parsed == {"ret_1d": 0.1, "range_pct": 0.02, "flag": 1.0}
    assert DENIED_FEATURE_COLUMNS.isdisjoint(parsed)

