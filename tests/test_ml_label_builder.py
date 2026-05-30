from __future__ import annotations

from ml_stock_selector.label_builder import build_labels, rank_label_from_pct
from tests.ml_fixtures import normalized_bars


def test_labels_are_long_format_and_use_configured_bases():
    labels = build_labels(normalized_bars(), [1, 2], label_bases=["from_close", "from_next_open"])

    assert {"from_close", "from_next_open"} == set(labels["label_base"])
    assert {"trade_date", "code", "horizon_d", "future_ret", "rank_label"}.issubset(labels.columns)
    assert labels[["trade_date", "code", "horizon_d", "label_base"]].duplicated().sum() == 0


def test_rank_label_is_head_heavy():
    assert rank_label_from_pct(0.995) == 4
    assert rank_label_from_pct(0.96) == 3
    assert rank_label_from_pct(0.91) == 2
    assert rank_label_from_pct(0.75) == 1
    assert rank_label_from_pct(0.30) == 0

