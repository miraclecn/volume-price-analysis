# ML Stock Selector Operating Notes

The `ml_stock_selector` subsystem reads `stock_bar_normalized_daily` from alpha-data and existing `vpa_*` tables from this repository. It writes only `ml_*` tables and artifacts under `outputs/ml/`.

Alpha-data owns industry normalization. When alpha-data cannot classify a
stock, it emits `industry_code = "UNKNOWN"` and `industry_name = "UNKNOWN"`
and marks `MISSING_INDUSTRY_CODE` in data-quality usability flags. VPA-ML keeps
UNKNOWN rows for feature mart, labels, training samples, and model scoring; it
does not repair or infer industry codes.

UNKNOWN is not the same as non-tradable. ST, suspension, liquidity,
`can_buy_next_open`, and trade-score filters still decide tradability. Portfolio
construction limits UNKNOWN exposure separately from normal industry
diversification: by default at most one UNKNOWN holding is allowed, and
`max_unknown_industry_names = 0` disables UNKNOWN entries. UNKNOWN holdings are
reported as a separate risk exposure in backtest metrics and reports.

Command order:

1. Run the VPA pipeline to produce `vpa_*` tables.
2. Validate alpha-data and VPA contracts.
3. Build feature mart and labels.
4. Train models and register artifacts.
5. Run batch prediction, scoring, portfolio construction, and T+1 backtest.
6. Generate daily signals from active registry artifacts.

Backtests use T-day information for decisions and execute no earlier than T+1.
