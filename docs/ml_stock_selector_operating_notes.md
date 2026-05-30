# ML Stock Selector Operating Notes

The `ml_stock_selector` subsystem reads `stock_bar_normalized_daily` from alpha-data and existing `vpa_*` tables from this repository. It writes only `ml_*` tables and artifacts under `outputs/ml/`.

Command order:

1. Run the VPA pipeline to produce `vpa_*` tables.
2. Validate alpha-data and VPA contracts.
3. Build feature mart and labels.
4. Train models and register artifacts.
5. Run batch prediction, scoring, portfolio construction, and T+1 backtest.
6. Generate daily signals from active registry artifacts.

Backtests use T-day information for decisions and execute no earlier than T+1.

