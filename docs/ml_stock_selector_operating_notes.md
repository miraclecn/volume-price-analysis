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

Portfolio v2 builds targets from two ranked pools. Hard filters are applied
first for BSE, ST, suspension, next-open buyability, ADV20, and v2 trade score.
The core pool is selected first as the strong-signal priority set; if it does
not fill `target_positions`, the candidate pool backfills the remaining slots.
When candidates are insufficient, the portfolio keeps cash instead of forcing a
full book.

For v2, `horizon_d = 5` means the model is trained to forecast a five-day
outcome; it is not a promise to hold exactly five days. The default holding
policy gives the signal time to work with `min_hold_days = 3`,
`target_hold_days = 5`, and `max_hold_days = 10`. Existing holdings are
evaluated before new buys, so the target portfolio is retained holdings plus
new entries, not a daily rebuild from zero.

Buy and sell thresholds intentionally differ. The default buy threshold is
`candidate_min_trade_score = 0.65`; the default score-exit threshold is
`sell_score_threshold = 0.45`. This hysteresis reduces churn. A holding can
exit through hard exits, risk exits, score exits, max-hold time exits, or
`not_candidate_after_target_days` after the target holding period. Hard and
risk exits can break `min_hold_days`; ordinary score deterioration cannot when
`allow_score_exit_before_min_hold = false`.

`core_pool` and `candidate_pool` are buy candidate pools only. Sell decisions
come from the holding policy. If `can_sell_next_open = false`, a sell signal is
recorded as `sell_blocked` and the stock remains held; the system must not
assume that exit filled.

`max_initial_entries` controls an empty or initial build. After holdings exist,
`max_new_entries_per_day` limits only stocks that are not already held, so
continuing holdings do not consume the daily new-entry budget. `hard_max_positions`
remains the absolute final holding cap.

Backtests and daily signals both use `construct_portfolio_targets_v2`. The
constructor attaches one `ml_portfolio_construction_diagnostics` row per decision
date with raw, hard-filtered, core, candidate, selected, and rejection counts.
Rejection counts are overlapping diagnostics: a row can contribute to multiple
reason counts, which is intentional for explaining sparse target days.
Backtest report output includes portfolio diagnostics metrics and a
`selected_count_distribution` CSV, so `wf_2020` can be checked for how often the
portfolio selected 0, 1, 4, 8, 12, or other target counts.

The fixed five-day baseline is separate from portfolio v2. Use
`abs_ranker_fixed_5d_risk_filter_v1` with
`[portfolio.fixed_5d_risk_filter]` to test a cleaner execution horizon:
T-day features generate a signal, T+1 open buys, and the default exit is after
five trading holding days. The Absolute Ranker is judged against the fixed
T+1-open to T+6-close return; the Risk Model is judged as an entry filter and,
when enabled, as the only early exit. This profile intentionally disables
`score_exit`, `not_candidate_after_target_days`, `trailing_profit_exit`, dynamic
sell scores, and candidate-pool/TopN drop exits. A holding with a lower current
absolute rank still remains until `holding_days >= 5` unless `risk_exit` fires.
If `can_sell_next_open = false`, the sell is blocked and the holding is carried
forward for another attempt.

Use `abs_ranker_fixed_5d_no_risk_exit_v1` with
`[portfolio.fixed_5d_no_risk_exit]` as the control. It keeps the same entry
filters and five-trading-day time exit but disables holding-period `risk_exit`,
so fold reports can compare whether `risk_exit` improves return or drawdown.
For fixed-horizon runs, inspect average/median/max holding days, buy/sell
counts, `risk_exit_count`, `time_exit_count`, `sell_blocked_count`, average
entry absolute/risk ranks, average cash ratio, and realized returns by exit
reason.

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
