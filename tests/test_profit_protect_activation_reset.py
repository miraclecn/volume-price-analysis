from __future__ import annotations

import pandas as pd

from scripts.reset_profit_protect_activation_live_sim import activation_buy_codes


def test_activation_buy_codes_returns_only_sorted_buy_codes() -> None:
    plan = pd.DataFrame(
        [
            {"code": "b", "side": "sell"},
            {"code": "c", "side": "buy"},
            {"code": "a", "side": "BUY"},
            {"code": "a", "side": "buy"},
        ]
    )

    assert activation_buy_codes(plan) == ["a", "c"]
