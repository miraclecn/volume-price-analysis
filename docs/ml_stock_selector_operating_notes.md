# ML Stock Selector Operating Notes

The `ml_stock_selector` subsystem reads `stock_bar_normalized_daily` from alpha-data and existing `vpa_*` tables from this repository. It writes only `ml_*` tables and artifacts under `outputs/ml/`.

The v2 ML path is an opt-in three-model design:

- `alpha_ranker` acts as the Absolute Ranker and learns `absolute_label` when v2 labels are enabled.
- `active_ranker` learns `active_label`, based on market and industry excess returns.
- `risk_model` learns `risk_label` as a binary downside-risk probability.

Industry metadata remains row metadata for reports and portfolio constraints. It must not enter training features in v2: enable `exclude_industry_metadata_from_features_json` for the feature mart and `feature_matrix_v2_deny_industry` for the matrix encoder. If alpha-data does not provide benchmark views, this repository computes market and industry benchmark returns locally from `stock_bar_normalized_daily`.

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

`config/ml_default.toml` currently enables all `[ml_v2]` flags, so the command
order below runs with the three-model path by default. To run legacy behavior,
turn off the relevant v2 flags before running the same command order.

Backtests use T-day information for decisions and execute no earlier than T+1.
