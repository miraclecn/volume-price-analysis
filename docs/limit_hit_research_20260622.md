# Limit-Hit Independent Research 2026-06-22

## Scope

This is an independent limit-hit research line. It reads the shared data foundation but writes its own database and reports under `outputs/limit_hit_research/`.

It must not write to existing research databases or live sim state.

Shared read-only inputs:

- `outputs/vpa.duckdb`
- `outputs/ml/ml_ret5_alpha_risk_20260619.duckdb`

Current primary research database:

- `outputs/limit_hit_research/limit_hit_research_close.duckdb`

Current primary report directory:

- `outputs/limit_hit_research/reports/limit_hit_close_v1_2020_2025/`

Older experimental database:

- `outputs/limit_hit_research/limit_hit_research.duckdb`
- This used earlier touch/old-label experiments and should not be used as the primary conclusion.

## Strategy Semantics

Signal date is T after close.

- Buy at T+1 open.
- Predict whether T+1 will close at limit-up.
- If T+1 does not close at limit-up, sell at T+2 open.
- If T+1 closes at limit-up, hold one extra trading day and sell at T+3 open.
- If the scheduled exit day is paused or lacks a usable bar, keep retrying on later trading days until sellable.

Limit success mode:

- `close`: `T+1 close >= T+1 limit_up * 0.999`

The earlier `touch` mode counted intraday limit touches, including failed boards. It produced misleading labels for this strategy and is not the primary mode.

## Current Selected Parameters

```text
limit_success_mode = close
limit_hit_extra_hold_days = 1
max_positions = 1
max_position_weight = 0.69
min_probability = 0.35
max_risk_prob = 0.10
entry_min_ret = 0.10
risk_weight = 0.35
slippage_bps = 10
commission_bps = 3
stamp_duty_bps = 5
candidate_min_ret = 0.02
min_adv20_amount = 20000000
min_amount = 20000000
exclude_st = true
exclude_bse = true
```

The selected parameters are also the defaults in `scripts/research_limit_hit_strategy.py` as of this note.

## Reproduction Command

```bash
python scripts/research_limit_hit_strategy.py \
  --out-db outputs/limit_hit_research/limit_hit_research_close.duckdb \
  --out-dir outputs/limit_hit_research/reports/limit_hit_close_v1_2020_2025 \
  --start-year 2020 \
  --end-year 2025
```

## Current Selected Backtest

Source table:

- `lh_selected_metrics`
- `lh_selected_yearly_metrics`
- `lh_selected_orders`
- `lh_selected_nav`

Overall 2020-2025 selected result:

```text
total_return   = 107.62%
annual_return  = 13.44%
max_drawdown   = -29.78%
sharpe         = 0.63
calmar         = 0.45
buy_count      = 59
```

Yearly selected result:

```text
2020  return -11.21%, maxDD -29.78%
2021  return +40.09%, maxDD -17.17%
2022  return  -9.18%, maxDD  -9.18%
2023  return +15.03%, maxDD -10.24%
2024  return +17.36%, maxDD  -6.77%
2025  return +36.12%, maxDD -18.33%
```

## Position Size Sweep

Source table:

- `lh_sweep_cap_metrics`

For `max_positions=1`, `min_probability=0.35`, `max_risk_prob=0.10`:

```text
cap 0.20: return 28.13%, maxDD  -9.52%
cap 0.30: return 43.80%, maxDD -14.07%
cap 0.40: return 59.31%, maxDD -18.34%
cap 0.50: return 76.00%, maxDD -22.45%
cap 0.55: return 84.15%, maxDD -24.43%
cap 0.58: return 89.63%, maxDD -25.58%
cap 0.60: return 92.77%, maxDD -26.36%
cap 0.65: return 101.00%, maxDD -28.26%
cap 0.68: return 106.70%, maxDD -29.41%
cap 0.69: return 107.62%, maxDD -29.78%
cap 0.70: return -22.63%, maxDD -30.12%
```

`0.69` is the highest tested position weight that stays under the 30% drawdown target at the baseline 10 bps slippage after adding `entry_min_ret >= 0.10`. It is close to the limit. `0.65` or `0.68` are safer high-return alternatives under the same 10 bps assumption.

## Cost Sensitivity

Source table:

- `lh_cost_sensitivity_metrics`
- `lh_robust_metrics`
- `lh_robust_yearly_metrics`
- `lh_robust_orders`
- `lh_robust_nav`

The selected 0.69 weight is not robust to higher slippage. At 20 bps, it breaches the drawdown limit and triggers the drawdown stop early.

```text
cap 0.45, 30 bps: return 50.79%, maxDD -21.97%
cap 0.50, 30 bps: return 56.43%, maxDD -24.18%
cap 0.55, 30 bps: return 61.82%, maxDD -26.24%
cap 0.58, 30 bps: return 65.09%, maxDD -27.52%
cap 0.60, 30 bps: return 67.10%, maxDD -28.31%
cap 0.62, 30 bps: return 69.56%, maxDD -29.11%
cap 0.65, 30 bps: return -23.93%, maxDD -30.37%
cap 0.69, 20 bps: return -23.76%, maxDD -30.81%
```

For a robust default under worse execution, `max_position_weight=0.62` is the current highest tested point that remains under 30% drawdown at 30 bps slippage. For the baseline research objective using 10 bps slippage, `0.69` remains the highest-return tested point.

Formal robust alternative:

```text
max_position_weight = 0.62
slippage_bps        = 30
total_return        = 69.56%
annual_return       = 9.54%
max_drawdown        = -29.11%
sharpe              = 0.52
calmar              = 0.33
buy_count           = 59
```

Yearly robust alternative:

```text
2020  return -13.70%, maxDD -29.11%
2021  return +31.81%, maxDD -16.81%
2022  return  -9.57%, maxDD  -9.57%
2023  return +13.00%, maxDD  -9.62%
2024  return +12.79%, maxDD  -6.70%
2025  return +29.32%, maxDD -16.67%
```

## Market Regime And Broader Board Sweep

Follow-up run:

```bash
python scripts/research_limit_hit_regime_sweep.py \
  --source-db outputs/limit_hit_research/limit_hit_research_close.duckdb \
  --shared-ml-db outputs/ml/ml_ret5_alpha_risk_20260619.duckdb \
  --out-dir outputs/limit_hit_research/reports/limit_hit_regime_sweep_2020_2025
```

Output files:

- `outputs/limit_hit_research/reports/limit_hit_regime_sweep_2020_2025/regime_sweep_metrics.csv`
- `outputs/limit_hit_research/reports/limit_hit_regime_sweep_2020_2025/regime_sweep_yearly_metrics.csv`
- `outputs/limit_hit_research/reports/limit_hit_regime_sweep_2020_2025/regime_sweep_trade_attribution.csv`

Key results:

```text
base_pos1_cap069                         total +107.62%, annual +13.20%, maxDD -29.78%, buys  59
base_pos1_cap062                         total  +96.24%, annual +12.12%, maxDD -27.11%, buys  59
mkt_score40_50_pos1_cap069               total  +71.13%, annual  +9.55%, maxDD -21.01%, buys  36
mkt_score40_50_pos1_cap062               total  +63.56%, annual  +8.71%, maxDD -19.04%, buys  36
broad_p030_r015_pos2_cap035              total  -29.27%, annual  -5.71%, maxDD -30.20%, buys  97
broad_p025_r015_pos2_cap030              total  -30.43%, annual  -5.97%, maxDD -30.43%, buys 119
broad_p020_r020_pos3_cap025              total  -30.47%, annual  -5.98%, maxDD -30.71%, buys  88
broad_p025_r015_pos2_cap030_mkt40_50     total  -22.56%, annual  -4.25%, maxDD -30.84%, buys 175
prev_up375_475_pos1_cap069               total  -31.25%, annual  -6.16%, maxDD -33.28%, buys  13
```

Interpretation:

- Simple market timing did not improve the limit-hit strategy. VPA `market_score` filters lowered drawdown but reduced annual return materially. `prev_up_ratio` timing was harmful in this setup.
- Broadening the board pool by lowering probability/risk thresholds and allowing 2-3 concurrent positions increased trade count but turned the strategy negative. The weaker candidates do not contain a scalable limit-hit edge.
- The current model has some ranking signal only in the highest-confidence tail, but that tail is too sparse to compete with the main `profit_protect` strategy.
- This is not yet true intraday board-hitting research. With daily bars, the strategy is closer to "T close decision, T+1 open relay, predict whether T+1 closes limit-up." Real 打板 requires intraday order/fill data or at least a separate close-at-limit / next-day-relay study.

Selected attribution:

```text
base_pos1_cap069: 59 trades, win 42.37%, next-day close-limit hit 30.51%, avg return on cost +2.28%
broad_p030_r015_pos2_cap035: 97 trades, win 41.24%, hit 18.56%, avg return on cost -0.91%
broad_p025_r015_pos2_cap030: 119 trades, win 37.82%, hit 17.65%, avg return on cost -0.93%
```

For `base_pos1_cap069`, trades selected when `prev_up_ratio <= 0.35` contributed most of the PnL, so the earlier assumption that weak market breadth should suppress board attempts is not supported by this sample. This is the opposite of the existing `prev_up` exposure rule.

## Caveats

- The selected baseline strategy is concentrated: at most one position, up to 69% of equity.
- The 30% drawdown constraint is met in the 10 bps baseline backtest, but with very little margin.
- Under stricter 30 bps slippage, use the robust `max_position_weight=0.62` alternative instead of 0.69.
- A 2024 Q1 smoke run with the default parameters is negative. Treat smoke as a pipeline test, not an investment conclusion.
- The current implementation retrains models and writes predictions, labels, orders, NAV, and metrics to the independent DB. It does not create a separate feature store or matrix cache.
- The parameter sweep implementation is currently ad hoc and not optimized; repeated sweeps rebuild the bar index each run.
- Do not promote this line to live sim. Current results are materially weaker than `mkt_tier_profit_protect`, and broadening the trade pool makes performance worse.

## Next Research Steps

- Add efficient reusable backtest indexing for larger parameter sweeps.
- Replace the current "next-day relay limit" target with a cleaner board taxonomy:
  - `sealed_today`: T close at limit-up.
  - `failed_board_today`: T high touches limit-up but close does not seal.
  - `next_day_premium`: T+1 open/high/close return after sealed board.
  - `second_board_success`: T+1 close at limit-up after T sealed board.
- Use market state as a conditioning feature first, not as a hand-written exposure rule, because the tested breadth rules were harmful.
- If true intraday 打板 is required, add intraday bars/order data; daily bars can only approximate close-sealed board relay.

## Board Taxonomy Follow-Up

Follow-up run:

```bash
python scripts/research_board_taxonomy.py \
  --shared-ml-db outputs/ml/ml_ret5_alpha_risk_20260619.duckdb \
  --out-dir outputs/limit_hit_research/reports/board_taxonomy_2020_2025 \
  --start-date 2020-01-01 \
  --end-date 2025-12-31 \
  --min-adv20-amount 10000000
```

Output files:

- `outputs/limit_hit_research/reports/board_taxonomy_2020_2025/event_summary.csv`
- `outputs/limit_hit_research/reports/board_taxonomy_2020_2025/sealed_relay_segments.csv`
- `outputs/limit_hit_research/reports/board_taxonomy_2020_2025/sealed_by_ret_turnover.csv`
- `outputs/limit_hit_research/reports/board_taxonomy_2020_2025/failed_board_segments.csv`
- `outputs/limit_hit_research/reports/board_taxonomy_2020_2025/relay_candidate_segments.csv`
- `outputs/limit_hit_research/reports/board_taxonomy_2020_2025/overnight_board_market_rules.csv`

Taxonomy definitions:

- `sealed_today`: T close is at T limit-up.
- `failed_board_today`: T high touches limit-up but T close does not seal.
- `near_board_not_sealed`: T close is near limit-up but not sealed.
- `board_next_open_ret_net`: approximate return from buying a sealed board at T close/limit and exiting T+1 open, net of exit slippage/tax assumptions.
- `relay_next2_open_ret_net`: return from buying T+1 open and exiting T+2 open, net of buy/sell cost assumptions.
- `second_board_success`: after T sealed board, T+1 also closes at limit-up.

Key result:

```text
sealed_today:          count 83,278, board_next_open_ret_net +2.18%, win 65.79%, relay_next2_open_ret_net -0.81%, second-board 21.36%
failed_board_today:    count 42,127, board_next_open_ret_net -1.75%, win 19.88%, relay_next2_open_ret_net -0.47%, second-board  6.24%
near_board_not_sealed: count 43,410, board_next_open_ret_net -0.78%, win 29.49%, relay_next2_open_ret_net -0.21%, second-board  3.94%
```

This changes the diagnosis:

- The positive edge is in **owning a board after it has sealed and carrying it overnight**.
- The existing model/backtest mostly studies **next-day relay from T+1 open**, where the broad unconditional expectation is negative.
- Failed boards are structurally bad overnight candidates.
- Therefore the poor annual return is expected: the current daily-bar strategy is looking at the wrong tradable moment for 打板.

Market/environment notes:

- Sealed-board overnight premium is positive across many market states, not only in broad strong markets.
- Strong count-based heat windows are visible. For example, when previous sealed-board count is `100-200`, sealed-board next-open net returns are among the strongest large-count segments.
- The only large-count segment where relay is clearly positive is narrow: `prev_up_ratio 0.4-0.5`, previous sealed count `100-200`, `unknown` limit band, count `1,110`, relay net mean `+2.22%`. This is too narrow to promote without a proper out-of-sample model.

Rough overnight board-market rule simulation:

- `all_full`, `heat_20_200`, and similar rules produce unrealistically large compounded returns because they assume intraday ability to buy the sealed-board set at close/limit and diversify across many boards.
- Treat `overnight_board_market_rules.csv` as a directional sanity check only, not as a deployable backtest.
- The useful takeaway is not the absolute return number; it is that board ownership after a confirmed seal has a much stronger expectation than next-day relay.

Practical implication for strategy design:

- Do not add the current limit-hit relay model to live sim.
- If we want real 打板, the next required data improvement is intraday board/fill data:
  - first seal time,
  - reopen count,
  - seal duration,
  - remaining order imbalance / queue proxy,
  - whether a live order could actually fill.
- If intraday data is unavailable, the feasible daily-bar path is a **sealed-board next-day premium / second-board model**, not an intraday 打板 model.

## Sealed-Board Overnight Model

Follow-up run:

```bash
python scripts/research_board_overnight_model.py \
  --shared-ml-db outputs/ml/ml_ret5_alpha_risk_20260619.duckdb \
  --out-dir outputs/limit_hit_research/reports/board_overnight_model_2020_2025 \
  --train-start 2015-01-05 \
  --start-date 2020-01-01 \
  --end-date 2025-12-31 \
  --min-adv20-amount 10000000
```

Output files:

- `outputs/limit_hit_research/reports/board_overnight_model_2020_2025/board_overnight_metrics.csv`
- `outputs/limit_hit_research/reports/board_overnight_model_2020_2025/board_overnight_yearly_metrics.csv`
- `outputs/limit_hit_research/reports/board_overnight_model_2020_2025/board_overnight_selection_diagnostics.csv`
- `outputs/limit_hit_research/reports/board_overnight_model_2020_2025/board_overnight_prediction_deciles.csv`

Research setup:

- Train walk-forward models on sealed boards only.
- Features use T-day observable daily data plus previous-day market board statistics.
- Target is `target_ret_net = T+1 open / T close - 1 - buy commission - sell slippage - stamp duty`.
- Backtest assumes the strategy can buy the sealed board at T close/limit and sell T+1 open.
- This is an optimistic upper-bound study, not a live-executable backtest.

Model diagnostics:

```text
top_pred 5 boards, total exposure 5%: annual +121.81%, maxDD -0.21%, trades 6,960
random 5 boards, total exposure 5%:   annual  +29.75%, maxDD -0.29%, trades 6,960
all sealed boards, total exposure 5%: annual  +29.25%, maxDD -0.18%, trades 78,820
bottom_pred 5 boards:                 annual   -2.88%, maxDD -17.26%, trades 6,960
```

Prediction deciles are monotonic:

```text
decile 1: realized -0.09%, win 44.95%, second-board 21.71%
decile 5: realized +1.59%, win 64.10%, second-board 17.24%
decile 8: realized +2.66%, win 72.98%, second-board 20.55%
decile 9: realized +3.43%, win 75.74%, second-board 24.59%
decile 10: realized +6.69%, win 87.93%, second-board 53.01%
```

Low-exposure variants:

```text
top5_w01_total05: total exposure 5%,  single name 1%, annual +121.81%, maxDD -0.21%
top5_w02_total10: total exposure 10%, single name 2%, annual +390.52%, maxDD -0.41%
top3_w03_total09: total exposure 9%,  single name 3%, annual +398.34%, maxDD -0.54%
top5_w04_total20: total exposure 20%, single name 4%, annual +2277.43%, maxDD -0.83%
```

These returns are too high to treat as deployable because the fill assumption is the entire problem in real 打板. The useful conclusions are:

1. The daily model can rank sealed boards; the top decile has much stronger next-open premium and second-board probability.
2. Position sizing should start tiny if this line is ever connected to real execution: `1%` per board, max `5%` total exposure is the first research-sized setting.
3. Market filters were less important than board quality ranking. The next gating variable should be **fill quality / seal quality**, not broad market breadth alone.
4. Without intraday fillability data, this remains an upper-bound research artifact and should not be connected to live sim.

## Intraday Data Availability Audit

Follow-up run:

```bash
python scripts/research_intraday_data_audit.py \
  --out-json outputs/limit_hit_research/reports/intraday_data_audit_20260622.json
```

Audited DuckDB files:

- `/home/nan/alpha-data-local/output/raw.duckdb`
- `/home/nan/alpha-data-local/output/research_source.duckdb`
- `/home/nan/alpha-data-local/output/pit_reference_staging.duckdb`
- `outputs/ml/ml_ret5_alpha_risk_20260619.duckdb`
- `outputs/vpa.duckdb`

Audit result:

```text
database_count = 5
daily_limit_hit_count = 11
daily_available_but_not_enough = true
has_intraday_execution_data = false
```

The available data includes daily OHLCV, daily limit-up/down prices, pause/ST/BSE flags, and daily open-lock flags such as `is_limit_up_open_lock`. It does not include credible intraday execution data.

Required missing fields for real 打板:

- first limit-up time,
- reopen / failed-seal count,
- seal duration,
- queue volume / queue amount at limit-up,
- bid/ask depth or Level-2 order book,
- auction imbalance,
- whether a limit-order could actually fill.

Temporary daily-only proxy:

- Continue using `sealed_today`, `failed_board_today`, `second_board_success`, and `board_next_open_ret_net` as upper-bound labels.
- Use the sealed-board overnight model only to rank historical sealed boards and estimate whether the idea is worth buying data for.
- Do not connect this model to live sim because live cannot know or fill a sealed board from daily bars alone.

Data acquisition priority:

1. Minute bars with exact high/limit touch timestamps are the minimum viable next step.
2. Tick or Level-2 data with limit-up queue and reopen information is required for a serious 打板 simulator.
3. Broker/order logs are needed to calibrate actual fillability and slippage after a signal.

## Fill Stress Test

Follow-up run:

```bash
python scripts/research_board_fill_stress.py \
  --predictions outputs/limit_hit_research/reports/board_overnight_model_2020_2025/board_overnight_predictions.csv \
  --out-dir outputs/limit_hit_research/reports/board_fill_stress_2020_2025
```

Output files:

- `outputs/limit_hit_research/reports/board_fill_stress_2020_2025/board_fill_stress_metrics.csv`
- `outputs/limit_hit_research/reports/board_fill_stress_2020_2025/board_fill_stress_yearly_metrics.csv`

Stress setup:

- Candidate set is the model's top 5 sealed boards per day.
- Base exposure remains conservative: single name `1%`, max total `5%`.
- `ideal` means the top N predicted candidates fill.
- `random` means N random names inside the top 5 fill.
- `adverse` means the worst realized N names inside the top 5 fill. This approximates queue adverse selection: easy-to-fill boards are more likely to be lower quality.

Key results:

```text
ideal_top5_fill5:               annual +121.81%, maxDD -0.21%, fills 6,960
ideal_top5_fill3:               annual  +71.04%, maxDD -0.18%, fills 4,176
ideal_top5_fill2:               annual  +47.29%, maxDD -0.11%, fills 2,784
ideal_top5_fill1:               annual  +23.50%, maxDD -0.14%, fills 1,392
random_top5_fill2:              annual  +37.91%, maxDD -0.16%, fills 2,784
adverse_top5_fill3:             annual  +36.73%, maxDD -0.24%, fills 4,176
adverse_top5_fill2:             annual  +15.12%, maxDD -0.32%, fills 2,784
adverse_top5_fill2_extra50bps:  annual  +12.26%, maxDD -0.59%, fills 2,784
adverse_top5_fill2_extra100bps: annual   +9.47%, maxDD -1.35%, fills 2,784
adverse_top5_fill1:             annual   +3.05%, maxDD -1.00%, fills 1,392
```

Interpretation:

- The strategy is highly sensitive to fill quality, but not instantly destroyed by partial fills.
- If at least 2 of the top 5 boards can be filled most days, even a harsh adverse-selection assumption remains positive in this daily upper-bound test.
- If only 1 board fills and it is consistently the worst realized board in the top 5, the edge is too small to justify deployment.
- Practical starting position remains `1%` per filled board, `5%` total board sleeve. Increase only after broker fill logs prove the adverse-fill case is too conservative.

Execution gate before any live integration:

- Require a historical fillability estimate for top-5 model signals.
- Minimum useful target: at least 2 filled boards per active day with realized fill quality no worse than the `adverse_top5_fill2_extra50bps` stress case.
- If fillability is closer to `adverse_top5_fill1`, do not trade this line.

## Board Execution Data Contract

Engineering gate added:

```bash
python scripts/run_board_execution_contract_check.py --db <candidate_intraday_or_execution.duckdb>
```

Current daily-only data fails this gate as expected:

```text
python scripts/run_board_execution_contract_check.py \
  --db /home/nan/alpha-data-local/output/research_source.duckdb

ValueError: Missing tables: board_intraday_events
```

Minimum required table:

```text
board_intraday_events
  trade_date
  code
  first_limit_time
  last_limit_time
  seal_duration_seconds
  reopen_count
  limit_up
  close
  is_close_sealed
```

Optional but practically required for live-grade simulation:

```text
board_order_book_snapshots
  trade_date, code, snapshot_time
  bid_price_1, bid_volume_1
  ask_price_1, ask_volume_1
  limit_queue_volume

board_order_fills
  trade_date, code, signal_time, order_time
  side, order_price, order_qty
  filled_qty, avg_fill_price, status
```

Use stricter checks before any production claim:

```bash
python scripts/run_board_execution_contract_check.py \
  --db <candidate.duckdb> \
  --require-order-book \
  --require-fills
```

This contract is intentionally separate from the existing daily ML/VPA data contract. Daily OHLCV can support upper-bound research and sealed-board ranking, but it cannot prove fillability or queue adverse selection.

Standard table DDL:

- `sql/create_board_execution_tables.sql`

CSV import entrypoint:

```bash
python scripts/import_board_execution_data.py \
  --db outputs/limit_hit_research/board_execution.duckdb \
  --events-csv <board_intraday_events.csv> \
  --source <data_vendor_or_broker>
```

Strict import when order book and fills are available:

```bash
python scripts/import_board_execution_data.py \
  --db outputs/limit_hit_research/board_execution.duckdb \
  --events-csv <board_intraday_events.csv> \
  --order-book-csv <board_order_book_snapshots.csv> \
  --fills-csv <board_order_fills.csv> \
  --require-order-book \
  --require-fills \
  --source <data_vendor_or_broker>
```

Import behavior:

- Executes the standard DDL first.
- Upserts by primary key.
- Adds `source` and `ingested_at`.
- Runs `board_execution_contract` after import.
- In non-strict mode, missing order book/fill tables are warnings.
- In strict mode, missing `--order-book-csv` or `--fills-csv` fails immediately.

## Fillability Gate

Follow-up run:

```bash
python scripts/research_board_fillability_gate.py \
  --predictions outputs/limit_hit_research/reports/board_overnight_model_2020_2025/board_overnight_predictions.csv \
  --board-db outputs/limit_hit_research/board_execution.duckdb \
  --out-dir outputs/limit_hit_research/reports/board_fillability_gate_2020_2025
```

Output files:

- `outputs/limit_hit_research/reports/board_fillability_gate_2020_2025/board_fillability_gate_summary.json`
- `outputs/limit_hit_research/reports/board_fillability_gate_2020_2025/board_fillability_gate_daily.csv`
- `outputs/limit_hit_research/reports/board_fillability_gate_2020_2025/board_fillability_gate_fills.csv`

Gate semantics:

- For each trade date, select the sealed-board overnight model's top 5 candidates by `pred_ret`, `pred_win_prob`, and `code`.
- Match real buy fills from `board_order_fills` by `trade_date` and `code`.
- Use `target_ret_net` as the realized next-open proxy, adjusted by `order_price / avg_fill_price` when actual fill price is available.
- Compare actual fills against an adverse benchmark: worst realized 2 names inside the top 5, with an extra 50 bps cost.
- Pass only if average filled boards per active day is at least 2 and realized fill quality is no worse than the adverse benchmark.

Current status:

```text
ok     = false
reason = missing board execution database: outputs/limit_hit_research/board_execution.duckdb
```

This is the expected result before broker/vendor execution logs exist. It protects live sim by making the sealed-board model non-deployable until true fillability is measured. The current primary live strategy remains `mkt_tier_profit_protect`; this research line must not be wired into live execution before the gate passes.

## Market Environment And Position Sizing Sweep

Follow-up run:

```bash
python scripts/research_board_market_position_sweep.py \
  --predictions outputs/limit_hit_research/reports/board_overnight_model_2020_2025/board_overnight_predictions.csv \
  --out-dir outputs/limit_hit_research/reports/board_market_position_sweep_2020_2025
```

Output files:

- `outputs/limit_hit_research/reports/board_market_position_sweep_2020_2025/board_market_position_metrics.csv`
- `outputs/limit_hit_research/reports/board_market_position_sweep_2020_2025/board_market_position_yearly_metrics.csv`
- `outputs/limit_hit_research/reports/board_market_position_sweep_2020_2025/board_market_position_regime_attribution.csv`
- `outputs/limit_hit_research/reports/board_market_position_sweep_2020_2025/board_market_position_manifest.json`

Experiment setup:

- Still uses the sealed-board overnight upper-bound assumption: buy confirmed sealed board at T close/limit, sell T+1 open.
- Candidate ranking is the existing `board_overnight_predictions.csv`.
- Base sleeve is top 5 boards, `1%` per board, max total board exposure `5%`.
- Market variables are previous-day fields only:
  - `prev_sealed_count`: previous-day sealed board count / board heat.
  - `prev_up_ratio`: previous-day market up-ratio.
- Tested variants include hard market filters, half-size outside heat windows, heat-scaled exposure, previous-up scaled exposure, prediction-confidence filters, and confidence-scaled exposure.

Key results:

```text
base_top5_w01_total05:            annual +121.81%, maxDD -0.21%, buys 6,960, avg exposure 1.00
confidence_scaled_top5:           annual +220.32%, maxDD -0.31%, buys 6,960, avg exposure 1.44
heat_conf_scaled_top5:            annual +174.82%, maxDD -0.31%, buys 6,960, avg exposure 1.22
pred_prob_ge_70_top5:             annual +120.66%, maxDD -0.21%, buys 6,804, avg exposure 1.00
pred_ret_ge_2pct_top5:            annual +120.26%, maxDD -0.21%, buys 6,829, avg exposure 1.00
heat_20_200_top5:                 annual +114.95%, maxDD -0.21%, buys 6,660, avg exposure 0.96
prevup_scaled_top5:               annual +109.79%, maxDD -0.16%, buys 6,960, avg exposure 0.92
heat_scaled_top5:                 annual +106.32%, maxDD -0.21%, buys 6,960, avg exposure 0.87
prevup_30_40_top5:                annual  +72.81%, maxDD -0.11%, buys 5,220, avg exposure 0.67
base_top3_w01_total03:            annual  +71.04%, maxDD -0.18%, buys 4,176, avg exposure 1.00
```

Environment attribution for the base top5 sleeve:

```text
prev_sealed_count bucket:
000-020: trades 240,  realized mean +4.86%, win 81.67%
020-040: trades 1,990, realized mean +5.51%, win 84.37%
040-060: trades 2,255, realized mean +6.23%, win 86.16%
060-100: trades 2,000, realized mean +6.92%, win 89.55%
100-200: trades 415,   realized mean +8.55%, win 93.25%
200+:    trades 60,    realized mean +9.53%, win 93.33%

prev_up_ratio bucket:
00-30:  trades 1,740, realized mean +6.05%, win 86.38%
30-40:  trades 1,080, realized mean +6.10%, win 85.28%
40-50:  trades 955,   realized mean +5.91%, win 86.18%
50-60:  trades 965,   realized mean +6.36%, win 87.88%
60-80:  trades 1,550, realized mean +6.75%, win 88.45%
80-100: trades 670,   realized mean +7.13%, win 87.46%
```

Interpretation:

- `prev_sealed_count` is the stronger market-environment signal. Higher board heat has clearly higher realized premium in this upper-bound dataset.
- `prev_up_ratio` is not a good hard filter. Weak previous-day breadth still has strongly positive top5 sealed-board premium, so filtering weak breadth mostly removes profitable trades.
- Simple heat scaling underperformed fixed top5 because it reduced exposure too much in common moderate-heat regimes. Heat is useful for risk limits and attribution, but not enough as a standalone sizing rule.
- Prediction-confidence scaling is the best simple sizing rule in this upper-bound run, but it increases average exposure from `5%` to about `7.2%` equivalent sleeve exposure. It should not be used live until the fillability gate passes.

Research recommendation:

- For any future real 打板 sleeve, start with fixed `1%` per filled board, max `5%` total exposure.
- Do not suppress trades solely because `prev_up_ratio` is weak.
- Track `prev_sealed_count` as a risk/state variable:
  - low heat: keep the default sleeve or reduce only if fill quality is also poor;
  - high heat: allow more opportunities only after real fills prove that high-heat boards remain fillable.
- If true fill logs pass the gate, test a conservative confidence-sizing rule next: keep single-board cap at `1%`, allow total sleeve to rise only from `5%` to `6-7%` on high-confidence days.

## Board Signal Diagnostics

Follow-up run:

```bash
python scripts/research_board_signal_diagnostics.py \
  --predictions outputs/limit_hit_research/reports/board_overnight_model_2020_2025/board_overnight_predictions.csv \
  --out-dir outputs/limit_hit_research/reports/board_signal_diagnostics_2020_2025 \
  --top-n 5 \
  --min-segment-rows 50
```

Output files:

- `outputs/limit_hit_research/reports/board_signal_diagnostics_2020_2025/board_signal_single_factor_segments.csv`
- `outputs/limit_hit_research/reports/board_signal_diagnostics_2020_2025/board_signal_pairwise_segments.csv`
- `outputs/limit_hit_research/reports/board_signal_diagnostics_2020_2025/board_signal_top_candidate_profile.csv`
- `outputs/limit_hit_research/reports/board_signal_diagnostics_2020_2025/board_signal_diagnostics_manifest.json`

Top candidate profile:

```text
all sealed boards:
  rows 78,820
  realized mean +2.15%
  win rate 65.33%
  second-board rate 22.57%
  turnover mean 10.61%
  median adv20 amount 187.5m

model top5 per day:
  rows 6,960
  realized mean +6.34%
  win rate 86.95%
  second-board rate 50.19%
  turnover mean 3.68%
  median adv20 amount 83.7m
```

This is the clearest current description of the discovered limit-up pattern:

- The model is not simply buying all boards. It strongly concentrates into low-turnover, lower-liquidity sealed boards.
- Low turnover is a major explanatory variable:
  - top5 `turn_q1`: realized mean `+7.21%`, win `91.48%`, second-board `59.40%`;
  - top5 `turn_q5`: realized mean `+4.01%`, win `71.10%`, second-board `22.05%`.
- Low ADV is also favorable:
  - top5 `adv_q1`: realized mean `+6.98%`, win `89.17%`, second-board `55.62%`;
  - top5 `adv_q5`: realized mean `+5.39%`, win `86.50%`, second-board `40.51%`.
- Board heat is monotonic inside the model's top candidates:
  - top5 `lt20` previous sealed count: realized mean `+4.89%`, second-board `33.90%`;
  - top5 `60-100`: realized mean `+6.90%`, second-board `57.08%`;
  - top5 `100-200`: realized mean `+8.54%`, second-board `63.08%`;
  - top5 `gt200`: realized mean `+9.53%`, second-board `66.67%`.
- The broad market up-ratio is weaker than board heat. It changes expected premium only modestly and should not be a primary entry filter.
- `limit_20pct` boards look strong in top5 (`+9.81%` mean), but sample size is only 89 trades. Treat this as a follow-up hypothesis, not a standalone rule.

Best pairwise cells among top5:

```text
ret1 gt15 + turnover q1:       rows 249, mean +12.29%, win 95.58%, second-board 69.08%
heat gt200 + turnover q1:      rows  52, mean  +9.77%, win 94.23%, second-board 73.08%
heat 100-200 + turnover q1:    rows 355, mean  +8.69%, win 94.08%, second-board 67.61%
heat 60-100 + turnover q1:     rows 1,484, mean +7.42%, win 92.12%, second-board 63.54%
pred_ret q5 + turnover q1:     rows 4,527, mean +7.33%, win 91.78%, second-board 60.75%
```

Weak pairwise cells are mostly high-turnover cells even when predicted return is high:

```text
pred_ret q5 + turnover q5: mean +4.38%, second-board 20.11%
pred_ret q5 + turnover q4: mean +4.05%, second-board 20.91%
gt15 ret1 + turnover q5:   mean +4.11%, second-board 16.82%
```

Current research interpretation:

- The limit-up law found so far is closer to **sealed board + low turnover/low ADV + high model rank + stronger board heat**, not broad-market strength.
- This matches the older main-strategy attribution that low ADV / reversal-like behavior carried much of the return.
- The practical risk is execution: the best cells are exactly where fillability may be hardest. Therefore the next real validation must use broker/order data, not more daily-bar return fitting.
- If fill logs pass the gate, the first live-grade candidate rule should be:
  - rank sealed boards by the overnight model;
  - prefer low-turnover candidates;
  - use board heat as a sleeve risk state;
  - cap at `1%` per filled board and `5%` total until real fills prove otherwise.

## Turnover-Based Fill Proxy Sweep

Follow-up run:

```bash
python scripts/research_board_fill_proxy_sweep.py \
  --predictions outputs/limit_hit_research/reports/board_overnight_model_2020_2025/board_overnight_predictions.csv \
  --out-dir outputs/limit_hit_research/reports/board_fill_proxy_sweep_2020_2025
```

Output files:

- `outputs/limit_hit_research/reports/board_fill_proxy_sweep_2020_2025/board_fill_proxy_metrics.csv`
- `outputs/limit_hit_research/reports/board_fill_proxy_sweep_2020_2025/board_fill_proxy_yearly_metrics.csv`
- `outputs/limit_hit_research/reports/board_fill_proxy_sweep_2020_2025/board_fill_proxy_bucket_attribution.csv`
- `outputs/limit_hit_research/reports/board_fill_proxy_sweep_2020_2025/board_fill_proxy_manifest.json`

Purpose:

- The best signal cells are low-turnover boards, which are likely harder to buy after sealing.
- This sweep applies a daily-bar fillability proxy by turnover bucket.
- It does not redistribute unfilled cash: each top5 candidate has a target `1%` weight, and expected exposure is `1% * fill_probability`.
- This is still not a substitute for broker fills, but it tests whether the edge survives a severe fill haircut.

Fill probability assumptions:

```text
optimistic:      turn_q1 50%, turn_q2 70%, turn_q3 80%, turn_q4 90%, turn_q5 95%
neutral:         turn_q1 25%, turn_q2 45%, turn_q3 60%, turn_q4 75%, turn_q5 85%
conservative:    turn_q1 10%, turn_q2 25%, turn_q3 40%, turn_q4 55%, turn_q5 70%
severe adverse:  turn_q1  5%, turn_q2 15%, turn_q3 40%, turn_q4 70%, turn_q5 90%
```

Key results:

```text
optimistic_turnover_fill:                  annual +57.01%, maxDD -0.10%, expected fills/day 2.98
neutral_turnover_fill:                     annual +29.66%, maxDD -0.07%, expected fills/day 1.80
neutral_turnover_fill_extra50bps:          annual +26.75%, maxDD -0.10%, expected fills/day 1.80
conservative_turnover_fill:                annual +14.07%, maxDD -0.07%, expected fills/day 0.98
conservative_turnover_fill_extra50bps:     annual +12.67%, maxDD -0.08%, expected fills/day 0.98
severe_adverse_turnover_fill:              annual +10.44%, maxDD -0.12%, expected fills/day 0.82
severe_adverse_turnover_fill_extra50bps:   annual  +9.31%, maxDD -0.13%, expected fills/day 0.82
```

Yearly result under the harshest listed proxy (`severe_adverse_turnover_fill_extra50bps`):

```text
2020 +6.42%
2021 +9.11%
2022 +8.78%
2023 +12.83%
2024 +10.66%
2025 +7.95%
```

Bucket contribution under neutral fill:

```text
turn_q1: attempts 4,672, expected fills 1,168.00, realized mean +7.21%
turn_q2: attempts 1,060, expected fills   477.00, realized mean +5.13%
turn_q3: attempts   543, expected fills   325.80, realized mean +4.46%
turn_q4: attempts   422, expected fills   316.50, realized mean +3.64%
turn_q5: attempts   263, expected fills   223.55, realized mean +4.01%
```

Interpretation:

- The edge survives large turnover-based fill haircuts in this proxy study.
- However, the neutral expected fill count is only `1.80` per active day, below the earlier live-grade gate of `2` real fills/day.
- The severe scenario is still positive, but expected fills/day is only `0.82`; this is not enough to justify deployment by itself.
- The practical conclusion is stronger than before: the signal is worth validating with real order/fill logs, but still should not be connected to live sim without those logs.

Working position rule after this proxy:

- Research sleeve: keep `1%` per attempted board, max 5 attempts.
- Live-grade sleeve: require real fills/day >= 2 before using the full 5% sleeve.
- If real fills/day is between 1 and 2, cap the sleeve below 5% or keep it paper-only.
- If real fills/day is below 1, do not trade this line even if daily-bar proxy remains positive.

## Fill-Aware Selection Sweep

Follow-up run:

```bash
python scripts/research_board_fill_aware_selection.py \
  --predictions outputs/limit_hit_research/reports/board_overnight_model_2020_2025/board_overnight_predictions.csv \
  --out-dir outputs/limit_hit_research/reports/board_fill_aware_selection_2020_2025
```

Output files:

- `outputs/limit_hit_research/reports/board_fill_aware_selection_2020_2025/board_fill_aware_selection_metrics.csv`
- `outputs/limit_hit_research/reports/board_fill_aware_selection_2020_2025/board_fill_aware_selection_yearly_metrics.csv`
- `outputs/limit_hit_research/reports/board_fill_aware_selection_2020_2025/board_fill_aware_selection_bucket_attribution.csv`
- `outputs/limit_hit_research/reports/board_fill_aware_selection_2020_2025/board_fill_aware_selection_manifest.json`

Question:

- If the best alpha boards are harder to fill, should selection still use pure `pred_ret`, or should it account for fill probability?

Selection policies tested:

```text
alpha_top:                 select top5 by pred_ret
fill_prob_top:             select easiest-to-fill names
expected_pred_ret:         select top5 by pred_ret * turnover_fill_probability
low_turnover_first:        select lowest turnover first
alpha_low_turnover_only:   select by pred_ret after dropping turnover q4/q5
```

Key results:

```text
neutral_expected_pred_ret:       annual +35.24%, maxDD -0.25%, expected fills/day 2.90, selected mean return +4.87%
neutral_alpha_top:               annual +29.66%, maxDD -0.07%, expected fills/day 1.80, selected mean return +6.34%
neutral_alpha_low_turnover_only: annual +26.97%, maxDD -0.07%, expected fills/day 1.63, selected mean return +6.30%

conservative_expected_pred_ret:  annual +22.81%, maxDD -0.29%, expected fills/day 2.60, selected mean return +3.55%
conservative_alpha_top:          annual +14.07%, maxDD -0.07%, expected fills/day 0.98, selected mean return +6.34%

severe_expected_pred_ret:        annual +27.49%, maxDD -0.51%, expected fills/day 3.64, selected mean return +2.83%
severe_alpha_top:                annual +10.44%, maxDD -0.12%, expected fills/day 0.82, selected mean return +6.34%
```

Interpretation:

- Pure alpha selection (`alpha_top`) maximizes selected return, but the expected fill count is too low under neutral/conservative/severe fill assumptions.
- `expected_pred_ret = pred_ret * fill_probability` gives up some selected return but clears the practical fill-count hurdle in all tested fill-proxy regimes.
- Selecting only the easiest-to-fill boards (`fill_prob_top`) is not good enough; it raises fills but selects too much weak alpha.
- Dropping high-turnover candidates hurts under the fill proxy because it further reduces fill count. This is useful as a risk-control idea only if real fills show high-turnover boards are actually poor after costs.

Current candidate rule after fill-aware selection:

```text
Paper / research:
  rank sealed boards by pred_ret * estimated_fill_probability
  attempt top5
  single-board target weight 1%
  max attempted sleeve 5%

Live-grade:
  use broker/order logs to replace estimated_fill_probability
  require >= 2 real fills/day before allowing a 5% sleeve
  keep alpha_top as an attribution benchmark, not the primary executable rule
```

This changes the practical direction: for real 打板, the objective is not just "find the highest expected board"; it is "maximize expected filled return". The daily model supplies the alpha side, but execution probability must become part of ranking before any live integration.

## Candidate Promotion Gate

Follow-up run:

```bash
python scripts/research_board_candidate_gate.py \
  --fillability-summary outputs/limit_hit_research/reports/board_fillability_gate_2020_2025/board_fillability_gate_summary.json \
  --selection-metrics outputs/limit_hit_research/reports/board_fill_aware_selection_2020_2025/board_fill_aware_selection_metrics.csv \
  --signal-profile outputs/limit_hit_research/reports/board_signal_diagnostics_2020_2025/board_signal_top_candidate_profile.csv \
  --out-json outputs/limit_hit_research/reports/board_candidate_strategy_manifest.json \
  --candidate-variant neutral_expected_pred_ret \
  --min-live-fills-per-day 2.0
```

Output:

- `outputs/limit_hit_research/reports/board_candidate_strategy_manifest.json`

Current manifest status:

```text
status = paper_only
candidate_variant = neutral_expected_pred_ret
ranking = pred_ret * estimated_fill_probability
single_name_weight = 1%
max_attempted_sleeve = 5%
proxy annual_return = +35.24%
proxy expected fills/day = 2.90
blocker = fillability gate failed: missing board execution database
```

Promotion rule:

- `paper_only`: research may continue, but live sim and production execution must not use the strategy.
- `live_candidate`: allowed only when:
  - fillability gate passes with real broker/vendor fill logs;
  - real fills/day is at least 2;
  - candidate annual return remains positive after real-fill calibration.

Current conclusion:

- The candidate strategy is good enough to justify collecting/importing execution logs.
- It is not good enough to connect to live sim yet because the strongest alpha depends on execution assumptions.
- `mkt_tier_profit_protect` remains the live main strategy. The board strategy is a separate paper-only candidate until `board_candidate_strategy_manifest.json` reports `status = live_candidate`.

## Execution Data Templates

Template export command:

```bash
python scripts/export_board_execution_templates.py \
  --out-dir outputs/limit_hit_research/board_execution_templates
```

Generated files:

- `outputs/limit_hit_research/board_execution_templates/board_intraday_events_template.csv`
- `outputs/limit_hit_research/board_execution_templates/board_order_book_snapshots_template.csv`
- `outputs/limit_hit_research/board_execution_templates/board_order_fills_template.csv`
- `outputs/limit_hit_research/board_execution_templates/README.md`

Data request document:

- `docs/board_execution_data_request_20260622.md`

These templates match `board_execution_contract` exactly. Replace the example rows with real vendor/broker data, then import:

```bash
python scripts/import_board_execution_data.py \
  --db outputs/limit_hit_research/board_execution.duckdb \
  --events-csv <board_intraday_events.csv> \
  --order-book-csv <board_order_book_snapshots.csv> \
  --fills-csv <board_order_fills.csv> \
  --require-order-book \
  --require-fills \
  --source <vendor_or_broker>
```

Then rerun:

```bash
python scripts/audit_board_execution_data.py
python scripts/research_board_fillability_gate.py
python scripts/research_board_candidate_gate.py
```

Expected state before real data:

```text
board_candidate_strategy_manifest.status = paper_only
```

Required state before any live sim integration:

```text
board_candidate_strategy_manifest.status = live_candidate
```

Hard live gate check:

```bash
python scripts/check_board_live_gate.py \
  --manifest outputs/limit_hit_research/reports/board_candidate_strategy_manifest.json
```

Current expected result:

```text
ok = false
status = paper_only
exit_code = 1
```

Any future implementation that attempts to wire the board strategy into live sim should run this check first and abort unless it returns `ok = true`.

Integration test coverage:

- `tests/test_board_research_gate_integration.py` builds a temporary `board_execution.duckdb` with synthetic top-candidate buy fills.
- It verifies `research_board_fillability_gate.py` passes when real fills cover at least 2 top candidates per active day.
- It verifies `research_board_candidate_gate.py` then promotes the candidate manifest to `status = live_candidate`.

Therefore the current `paper_only` state is caused by missing real execution data, not by an untested promotion path.

## Profit-Protect Overlay Study

Follow-up run:

```bash
python scripts/research_board_overlay_profit_protect.py \
  --main-nav outputs/ml/reports/profit_protect_continuous_wf_2020_2025_20260622/continuous_nav.csv \
  --out-dir outputs/limit_hit_research/reports/board_overlay_profit_protect_2020_2025
```

Output files:

- `outputs/limit_hit_research/reports/board_overlay_profit_protect_2020_2025/board_overlay_metrics.csv`
- `outputs/limit_hit_research/reports/board_overlay_profit_protect_2020_2025/board_overlay_yearly_metrics.csv`
- `outputs/limit_hit_research/reports/board_overlay_profit_protect_2020_2025/board_overlay_manifest.json`

Overlay semantics:

- Main leg is the existing continuous `mkt_tier_profit_protect` NAV.
- Board leg uses paper/proxy daily returns from fill-aware selection experiments.
- Combined daily return is `main_return + board_return * scale`.
- `scale10` means use the board candidate at its researched `1% * top5 = 5% attempted sleeve`.
- `scale05` means half the board sleeve.
- This is research-only. The board leg remains paper/proxy until real fillability passes.

Key combined results:

```text
main_only:                            annual +51.66%, maxDD -18.32%
board_neutral_expected_scale05:       annual +75.15%, maxDD -17.20%
board_neutral_expected_scale10:       annual +102.26%, maxDD -16.80%
board_conservative_expected_scale10:  annual +84.47%, maxDD -17.04%
board_severe_expected_scale10:        annual +91.15%, maxDD -16.72%
```

Yearly comparison:

```text
main_only:
  2020 +32.80%, 2021 +58.96%, 2022 +60.27%, 2023 +14.86%, 2024 +84.74%, 2025 +58.56%

board_neutral_expected_scale10:
  2020 +73.40%, 2021 +109.94%, 2022 +112.23%, 2023 +52.65%, 2024 +153.16%, 2025 +112.55%

board_conservative_expected_scale10:
  2020 +57.61%, 2021 +94.47%, 2022 +91.99%, 2023 +42.31%, 2024 +125.82%, 2025 +93.68%

board_severe_expected_scale10:
  2020 +62.35%, 2021 +102.73%, 2022 +97.46%, 2023 +48.56%, 2024 +134.00%, 2025 +100.52%
```

Interpretation:

- The board candidate has meaningful diversification/return potential against the main profit-protect strategy in paper/proxy form.
- The improvement is largest in 2023, where the main strategy had weak opportunity capture.
- The overlay does not increase historical max drawdown in this proxy study; it slightly reduces it because the board leg's proxy drawdown is tiny and not highly synchronized with the main leg.
- This does not override the live gate. The correct conclusion is: if real fillability passes, a small board sleeve could materially improve the current strategy stack. Until then it remains a paper overlay only.

Overlay robustness diagnostics:

```bash
python scripts/research_board_overlay_diagnostics.py \
  --overlay-dir outputs/limit_hit_research/reports/board_overlay_profit_protect_2020_2025 \
  --variant board_neutral_expected_scale10
```

Key diagnostics for `board_neutral_expected_scale10`:

```text
main_board_corr = 0.13
board_mean_return_on_main_down_days = +0.10% per day
both_down_rate = 3.71%
board_sum_return_share = 39.18%
worst_main_day_return = -13.22%
worst_combined_day_return = -13.04%
worst_board_day_return = -0.25%
```

Yearly board contribution is positive every year:

```text
2020 board return sum +25.98%
2021 board return sum +27.10%
2022 board return sum +27.26%
2023 board return sum +27.63%
2024 board return sum +30.59%
2025 board return sum +28.54%
```

Worst-day inspection does not show the board leg amplifying the main strategy's large loss days. On the worst combined day, 2025-04-07, main return was `-13.22%` while the board leg proxy contributed `+0.18%`.

## Decision Summary Artifact

Summary command:

```bash
python scripts/summarize_board_research.py \
  --candidate-manifest outputs/limit_hit_research/reports/board_candidate_strategy_manifest.json \
  --overlay-metrics outputs/limit_hit_research/reports/board_overlay_profit_protect_2020_2025/board_overlay_metrics.csv \
  --out-dir outputs/limit_hit_research/reports/board_research_summary
```

Outputs:

- `outputs/limit_hit_research/reports/board_research_summary/board_research_summary.md`
- `outputs/limit_hit_research/reports/board_research_summary/board_research_summary.json`

Current decision summary:

```text
Status: paper_only
Decision: continue paper research and collect/import real board execution fills
Candidate: neutral_expected_pred_ret
Ranking: pred_ret * estimated_fill_probability
Candidate proxy annual return: +35.24%
Candidate expected fills/day: 2.90
Main-only annual/maxDD: +51.66% / -18.32%
Best overlay annual/maxDD: +102.26% / -16.80%
Blocker: missing board execution database
```

Use this summary as the first file to inspect before making future board-strategy decisions. The detailed CSVs remain the audit trail.

## Downstream Refresh Command

To refresh all downstream board research artifacts from the existing sealed-board overnight predictions and current profit-protect NAV:

```bash
python scripts/run_board_research_refresh.py
```

Dry-run command:

```bash
python scripts/run_board_research_refresh.py --dry-run
```

This runs, in order:

1. `research_board_fillability_gate.py`
2. `research_board_signal_diagnostics.py`
3. `research_board_fill_proxy_sweep.py`
4. `research_board_fill_aware_selection.py`
5. `research_board_candidate_gate.py`
6. `research_board_overlay_profit_protect.py`
7. `research_board_overlay_diagnostics.py`
8. `summarize_board_research.py`

It does not retrain the sealed-board overnight model and does not write to live sim. If real `board_execution.duckdb` data is later imported, rerun this command to update the fillability gate, candidate manifest, overlay diagnostics, and summary.

## Execution Data Coverage Audit

After importing real execution data, run:

```bash
python scripts/audit_board_execution_data.py \
  --predictions outputs/limit_hit_research/reports/board_overnight_model_2020_2025/board_overnight_predictions.csv \
  --board-db outputs/limit_hit_research/board_execution.duckdb \
  --out-json outputs/limit_hit_research/reports/board_execution_data_audit.json
```

Current output before real data:

```text
ok = false
reason = missing board execution database: outputs/limit_hit_research/board_execution.duckdb
```

When real data exists, this audit reports:

- contract status and missing tables/columns;
- event coverage against model top5 signals;
- order-book snapshot coverage against model top5 signals;
- buy fill coverage and average matched fills per active day;
- days with at least 2 matched fills, which is the current live-candidate threshold.
- `missing_top_sample`, up to 20 unmatched top5 `trade_date/code` pairs per table, so data gaps can be traced directly.
