from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.research_board_overlay_profit_protect import combine_returns, load_nav_returns


def test_load_nav_returns_can_use_first_return_from_initial_nav(tmp_path: Path) -> None:
    path = tmp_path / "board_nav.csv"
    pd.DataFrame(
        [
            {"trade_date": "2024-01-02", "nav": 1_010_000.0},
            {"trade_date": "2024-01-03", "nav": 1_020_100.0},
        ]
    ).to_csv(path, index=False)

    out = load_nav_returns(path, date_col="trade_date", nav_col="nav", first_return_from_initial=True)

    assert out["daily_return"].tolist() == pytest.approx([0.01, 0.01])


def test_combine_returns_adds_board_overlay_and_fills_missing_dates_with_zero() -> None:
    main = pd.DataFrame(
        [
            {"trade_date": "2024-01-02", "nav": 1_000_000.0, "daily_return": 0.01},
            {"trade_date": "2024-01-03", "nav": 1_010_000.0, "daily_return": -0.02},
        ]
    )
    board = pd.DataFrame(
        [
            {"trade_date": "2024-01-02", "nav": 1_000_000.0, "daily_return": 0.005},
        ]
    )

    combined = combine_returns(main, board, board_scale=2.0, initial_nav=100_000.0)

    assert combined["combined_return"].tolist() == pytest.approx([0.02, -0.02])
    assert combined["nav"].tolist() == pytest.approx([102_000.0, 99_960.0])
