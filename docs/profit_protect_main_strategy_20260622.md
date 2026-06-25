# Profit Protect 主策略归档

归档日期: 2026-06-22

状态: 暂定为当前主研究策略。该策略已完成研究脚本回测，但尚未迁移为 live sim / production 默认策略。

## 策略身份

- 主策略名: `mkt_tier_profit_protect`
- 基准策略: `mkt_tier_prev_up375_475`
- 研究脚本: `scripts/research_risk_controls.py`
- 主要输出目录:
  - `outputs/ml/reports/not_candidate_strategy_variants_r120_20260622/`
  - `outputs/ml/reports/not_candidate_strategy_variants_r120_2026ytd_20260622/`
- 核心结果文件:
  - `strategy_variant_summary_ranked.csv`
  - `strategy_variant_yearly.csv`
  - `exit_reason_summary.csv`
  - `risk_control_orders.csv`

## 模型与数据口径

- ML DB: `outputs/ml/ml_ret5_alpha_risk_20260619.duckdb`
- run_id: `wf_v2_ret5_fund_fixed_a160_r120_20260621`
- score_version: `v2_alpha_ret5d_fund_fixed_a160_r120_20260621`
- 模型轮次: alpha 固定 160 轮，risk 固定 120 轮。
- 训练口径: 固定轮次 walk-forward；每个测试年使用测试年前历史训练，不使用错误的测试年 early stop。
- 回测区间: 2020-2025；另跑 2026 YTD 稳健性参考。
- 初始资金: 1,000,000。
- 成本: 10 bps 滑点，3 bps 佣金，5 bps 印花税。
- 执行: T 日决策，T+1 next open 成交。

## 入选与基础退出规则

评分:

```text
trade_score_v2 = 0.85 * absolute_rank_pct + 0.15 * low_adv_score
low_adv_score = 1 - full_prediction_pool_adv_pct
```

基础组合约束:

- target positions: 12
- hard max positions: 15
- max initial entries: 12
- max new entries per day: 4
- min ADV20: 10,000,000
- 排除 ST、停牌、北交所、不可次日开盘买入标的
- candidate: `trade_score_v2 >= 0.75`，`absolute_rank_pct >= 0.70`，`risk_rank_pct <= 0.55`
- core: `trade_score_v2 >= 0.75`，`absolute_rank_pct >= 0.75`，`risk_rank_pct <= 0.45`

市场仓位:

- 前一日可交易股票上涨比例 `< 0.375`: 目标仓位降为 0
- 前一日可交易股票上涨比例 `< 0.475`: 目标仓位降为 0.5
- 其他情况: 目标仓位 1.0

基础退出:

- risk exit: `risk_rank_pct >= 0.75` 或 `risk_prob >= 0.60`
- score exit: `trade_score_v2 < 0.35`，且满足最小持仓天数
- not candidate exit: 持仓满 5 个交易日后，不在 candidate pool 则退出
- time exit: 持仓满 10 个交易日强制退出

## Profit Protect 规则

目的: 处理“曾有浮盈但回吐成亏损/小盈利”的持仓管理问题，避免盈利票拖成亏损。

触发条件:

- 持仓交易日 `>= 3`
- 持仓期间最高价相对买入价曾达到 `+3%`
- 当前收盘价相对买入价 `<= +0.5%`

执行方式:

- 触发后生成 `profit_protect_exit` 卖出信号，次一交易日开盘成交。
- 若该持仓已经触发 risk / score / time / hard exit，则保留原退出原因，不用 profit protect 覆盖。
- 若原退出原因是 `not_candidate_after_target_days`，可被 `profit_protect_exit` 覆盖。

注意: 当前实现位于研究脚本的 target 后处理逻辑中，不应直接认为 live sim 已启用该规则。

## 2020-2025 回测结果

总览:

| 策略 | 总收益 | 年化收益 | 最大回撤 | 交易数 | 胜率 | 平均盈利 | 平均亏损 | 亏/盈比 | Profit Factor |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| base `mkt_tier_prev_up375_475` | +637.54% | 39.52% | -22.52% | 2342 | 51.28% | +6.92% | -4.84% | 0.70 | 1.50 |
| `mkt_tier_profit_protect` | +916.60% | 47.18% | -17.61% | 2417 | 48.61% | +7.00% | -4.39% | 0.63 | 1.51 |
| `mkt_tier_combo_h8` | +857.33% | 45.72% | -17.61% | 2403 | 49.65% | +6.82% | -4.46% | 0.65 | 1.51 |

年度结果:

| 年份 | 收益 | 最大回撤 | 交易数 | 胜率 | 平均盈利 | 平均亏损 | 亏/盈比 | Profit Factor |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2020 | +27.03% | -13.20% | 380 | 50.00% | +6.59% | -5.09% | 0.77 | 1.29 |
| 2021 | +54.28% | -14.53% | 421 | 47.03% | +7.36% | -4.37% | 0.59 | 1.50 |
| 2022 | +73.90% | -15.11% | 398 | 46.48% | +7.42% | -4.14% | 0.56 | 1.56 |
| 2023 | +8.78% | -9.63% | 399 | 44.86% | +4.78% | -3.26% | 0.68 | 1.19 |
| 2024 | +85.38% | -16.97% | 414 | 47.83% | +9.97% | -5.66% | 0.57 | 1.61 |
| 2025 | +47.92% | -17.61% | 405 | 55.56% | +5.82% | -3.86% | 0.66 | 1.88 |

## 2026 YTD 参考

| 策略 | 收益 | 最大回撤 | 交易数 | 胜率 | 平均盈利 | 平均亏损 | 亏/盈比 | Profit Factor |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| base `mkt_tier_prev_up375_475` | +11.58% | -13.71% | 175 | 48.57% | +5.45% | -4.73% | 0.87 | 1.09 |
| `mkt_tier_profit_protect` | +14.71% | -11.13% | 182 | 51.65% | +5.46% | -4.58% | 0.84 | 1.28 |
| `mkt_tier_combo_h8` | +15.86% | -10.51% | 177 | 51.98% | +5.68% | -4.86% | 0.85 | 1.27 |

2026 YTD 中 `combo_h8` 略优，但 2020-2025 主样本里 `profit_protect` 的总收益、年化收益和策略简洁性更好。因此当前暂以 `profit_protect` 作为主策略，`combo_h8` 作为候选观察策略。

## 研究结论

1. 当前策略不是纯低波防守策略，而是依赖横截面波动和短线扩张机会的波动收益捕捉策略。
2. 2023 收益和回撤都低，主要因为市场横截面波动较低、大赢家数量少；不是仓位不足。
3. `not_candidate` 全量延长持有并不稳健，单独 `grace_lowadv` 收益不如主策略。
4. `profit_protect` 能显著改善 2020-2025 的年化收益和最大回撤，并降低平均亏损/平均盈利比例。
5. 后续如要进入 live sim，应把 `profit_protect` 从研究脚本迁移到正式 `HoldingPolicy` / portfolio constructor，并补单元测试与 replay smoke。

## 当前注意事项

- `mkt_tier_profit_protect` 目前是研究脚本策略，不是生产默认策略。
- 新规则是 target 后处理，`risk_control_orders.csv` 是退出统计可信来源；`risk_control_diagnostics.csv` 的退出计数不适合分析新增规则。
- 该策略应继续使用 r120 固定轮次模型作为当前主候选，不应与旧 early-stop 错误验证结果混用。
- 若未来修改模型、标签、仓位或成本参数，必须重新生成同口径对比。
