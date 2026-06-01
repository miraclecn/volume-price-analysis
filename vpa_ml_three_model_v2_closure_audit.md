# VPA-ML Three-Model v2 Closure Audit

Date: 2026-06-01

Objective: execute `vpa_ml_three_model_v2_closure_tasks.md` and verify DoD against code/test/data evidence.

## DoD Checklist

1. 2015-01-05 样本进入训练 / walk-forward / 回测  
Status: PASS  
Evidence:
- `config/ml_walkforward.toml` all folds use `train_start = "2015-01-05"`.
- SQL: `ml_feature_mart_daily` range is `2015-01-05 ~ 2026-05-29` (11,216,101 rows).
- Walk-forward runner reads fold train/test boundaries from config: `ml_stock_selector/backtest/walkforward.py`.

2. 北交所不进入训练样本  
Status: PASS  
Evidence:
- `build_training_samples(..., exclude_bse=True)` implemented in `ml_stock_selector/sample_builder.py`.
- `train_ml_models.py` and walk-forward call path pass universe exclusion.
- Tests: `tests/test_training_samples_exclude_bse.py`, `tests/test_universe_filter.py`.

3. 北交所不进入 v2 预测 / 组合 / 回测 / 日信号  
Status: PASS  
Evidence:
- Batch predict + daily signal both call universe filter.
- SQL (run `v2_closure_20260601`): prediction BJ rows = `0`; portfolio target BJ rows = `0`.
- SQL (daily `2026-05-29`): BJ rows = `0`.

4. Walk-forward 每个 fold 训练 absolute/active/risk 三模型  
Status: PASS  
Evidence:
- `run_walkforward_experiment` trains `train_alpha_ranker` + `train_active_ranker` + `train_risk_model`.
- Test: `tests/test_walkforward.py`.

5. Fold 模型注册但不激活 production active  
Status: PASS  
Evidence:
- Walk-forward uses `register_model(...)` only; no `activate_model(...)` in walk-forward path.
- Activation remains in production training script (`scripts/train_ml_models.py`).

6. Batch predict 支持 run_id / fold_id / date range / 三模型 model_id  
Status: PASS  
Evidence:
- CLI args in `scripts/run_ml_batch_predict.py`: `--run-id --fold-id --start-date --end-date --model-id`.

7. 预测表可追溯三模型分数与 trade_score_v2  
Status: PASS  
Evidence:
- `ml_predictions_daily` contains `absolute_score/active_score/risk_prob/trade_score_v2`.
- Also stores lineage fields: `absolute_model_id/active_model_id/risk_model_id`.
- SQL (`run_id=v2_closure_20260601`): `trade_score_v2` null rows = `0`.

8. 回测候选合并 tradeability metadata  
Status: PASS  
Evidence:
- Batch predict merges tradeability fields into prediction rows.
- Backtest script has metadata fallback join from `ml_tradeability_daily` when missing/null.

9. ST/停牌/可买/流动性/行业/UNKNOWN/BSE 约束生效  
Status: PASS  
Evidence:
- `apply_hard_filters` covers ST/停牌/可买/流动性。
- portfolio constructor enforces行业与 UNKNOWN 约束。
- walk-forward now builds fold targets through `construct_portfolio_targets_v2`.
- universe filter enforces BSE exclusion.
- Tests: `tests/test_portfolio_constructor.py`, `tests/test_ml_unknown_industry.py`, universe tests.

10. portfolio v2 使用配置阈值（非硬编码 `-999`）  
Status: PASS  
Evidence:
- `scripts/run_ml_backtest.py` builds `PortfolioConstraints` from `config.portfolio` + `config.ml_v2`.

11. 回测产出 fold/run 级 metrics  
Status: PASS  
Evidence:
- `ml_backtest_metrics` includes `run_id/fold_id/score_version`.
- SQL (`run_id=v2_closure_20260601`): metrics rows = `2`.

12. daily signal 只用 production active 三模型  
Status: PASS  
Evidence:
- `generate_daily_signal(..., use_v2=True)` loads active models via `load_active_model`.
- Active registry query confirms single active per model type.

13. 行业信息不进入 feature matrix  
Status: PASS  
Evidence:
- `deny_industry` path implemented in `ml_stock_selector/feature_matrix.py`.
- Test: `tests/test_no_industry_training_features.py`.

14. 所有新增测试通过  
Status: PASS  
Evidence:
- Acceptance suite run: `38 passed` (Phase 7 command set).
- Additional regression runs after final fixes passed (`tests/test_daily_signal.py`, `tests/test_walkforward.py`, etc.).

## Notes

- Added schema migration guard in `ml_stock_selector/storage.py` for older local DB files.
- Added formal walk-forward config `config/ml_walkforward.toml`.
