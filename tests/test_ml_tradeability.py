from __future__ import annotations

from ml_stock_selector.tradeability import build_tradeability_mart
from tests.ml_fixtures import normalized_bars


def test_tradeability_uses_next_open_and_rejects_paused_or_limits():
    bars = normalized_bars()
    bars.loc[(bars["code"] == "000001.SZ") & (bars["trade_date"] == "2024-01-03"), "open"] = 11.0
    bars.loc[(bars["code"] == "000001.SZ") & (bars["trade_date"] == "2024-01-03"), "limit_up"] = 11.0
    mart = build_tradeability_mart(bars, adv_window=2)

    row = mart[(mart["code"] == "000001.SZ") & (mart["trade_date"] == "2024-01-02")].iloc[0]
    assert row["next_trade_date"] == "2024-01-03"
    assert row["can_buy_next_open"] is False

    last = mart[(mart["code"] == "000001.SZ") & (mart["trade_date"] == "2024-01-08")].iloc[0]
    assert last["can_buy_next_open"] is False
    assert last["can_sell_next_open"] is False

