# VPA-ML Refactor Baseline 202606

This baseline freezes the current live-simulation lane before the run/fold
identity refactor expands further.

## Code Baseline

- Branch point: `771375c Add files via upload`
- Refactor branch: `feature/reproducible-live-refactor`
- Primary plan: `docs/vpa_ml_refactor_plan.md`

## Live Simulation Model Lane

Current live simulation uses the archived preferred score lane implemented in
`ml_stock_selector.serving.live_sim`.

- `score_version`: `preferred_adv10m_fulladv015_top12`
- `account_id`: `preferred_adv10m_paper`
- `portfolio_id`: `preferred_adv10m_fulladv015_top12`
- Initial cash: `300000.0`
- Target positions: `12`

## Execution Parameters

- Execution price: `next_open`
- Slippage: `5.0` bps
- Commission: `3.0` bps
- Stamp duty: `5.0` bps
- Fractional shares: disabled
- A-share lot size: `100`

## Portfolio Parameters

- Target positions: `12`
- Hard max positions: `15`
- Max initial entries: `12`
- Max new entries per day: `4`
- Minimum ADV20 amount: `10000000.0`
- Candidate minimum trade score: `0.75`
- Core minimum trade score: `0.75`
- Candidate absolute minimum rank pct: `0.70`
- Candidate active minimum rank pct: `0.70`
- Candidate risk max rank pct: `0.65`
- Core absolute minimum rank pct: `0.75`
- Core active minimum rank pct: `0.65`
- Core risk max rank pct: `0.55`
- BSE excluded: `true`

## Holding Policy

- Minimum hold days: `3`
- Target hold days: `5`
- Maximum hold days: `10`
- Sell score threshold: `0.45`
- Risk exit rank pct: `0.85`
- Risk exit probability: `0.70`
- Sell if no longer candidate after target days: `true`
- Force exit after max hold days: `true`
- Allow score exit before min hold: `false`

## Reproducibility Guardrails Added In This Batch

- Backtest outputs are scoped by `run_id`, `fold_id`, `strategy_id`, and
  `score_version`.
- Portfolio targets are scoped by `run_id`, `fold_id`, `portfolio_id`, and
  `score_version`.
- Fixed-horizon backtests keep the original walk-forward `fold_id`; they no
  longer store the strategy id in the fold field.
- LightGBM fold-cache training writes `*.params.json` beside each model artifact.
- `live_sim_reproducibility_snapshot()` exports the active live-sim model lane
  and execution/portfolio parameters as plain data.

## Known Boundaries

- This is not a full `RunContext` implementation yet.
- Existing legacy target writes without run metadata are isolated under
  `run_id = legacy`, `score_version = legacy`.
- Production activation and model bundles remain future work.
- External DuckDB contract tests are not part of this baseline unless
  `VPA_RUN_EXTERNAL_DUCKDB_TESTS=1` is enabled.
