在 VPA-ML 的 portfolio/backtest/daily_signal 中实现 holding-aware sell mechanism，使三模型 5日/10日收益预测与实际持仓周期一致，避免每日重新选股导致平均持仓只有 1~2 天。

背景：
当前 wf_2020 回测结果显示：

* 平均持仓 1.56 个交易日；
* 中位数 1 个交易日；
* 持仓段数 1003；
* 大部分持仓隔日或短线退出。
  这说明当前组合逻辑接近“每日重新选 TopN，没进目标池就卖出”，与 horizon_d=5/10 的模型预测目标不匹配。需要新增持仓感知组合构建和卖出机制。

目标：

1. target portfolio = retained holdings + new entries，不再每日从零重选。
2. 买入后至少持有 min_hold_days，除非触发 hard exit 或 risk exit。
3. 达到 target_hold_days 后，如果不再属于 candidate_pool，可卖出。
4. 超过 max_hold_days 后强制退出或进入强制评估。
5. trade_score_v2 明显恶化时卖出，但卖出阈值低于买入阈值，形成 hysteresis。
6. risk_rank_pct / risk_prob 显著恶化时卖出。
7. can_sell_next_open=false 时不能假设卖出成功，应保留持仓并记录 sell_blocked。
8. max_new_entries_per_day 只限制新增，不限制保留持仓。
9. 回测和 daily signal 使用同一套持仓感知逻辑。

请实现以下任务：

Task 1: 增加 holding/exit 配置

修改：

* config/ml_default.toml
* config/ml_walkforward.toml
* ml_stock_selector/config.py
* ml_stock_selector/portfolio/constraints.py
* tests/test_ml_config.py
* tests/test_portfolio_constructor_v2.py

新增配置：

[portfolio.v2.holding]
min_hold_days = 3
target_hold_days = 5
max_hold_days = 10

[portfolio.v2.exit]
sell_score_threshold = 0.45
risk_exit_rank_pct = 0.85
risk_exit_prob = 0.70
sell_if_not_candidate_after_target_days = true
force_exit_after_max_hold_days = true
allow_score_exit_before_min_hold = false

规则：

* horizon_d=5 默认 min_hold_days=3、target_hold_days=5、max_hold_days=10。
* horizon_d=10 可后续配置 min_hold_days=5、target_hold_days=10、max_hold_days=15。
* sell_score_threshold 必须低于 candidate_min_trade_score，形成买卖迟滞。
* allow_score_exit_before_min_hold=false 时，未满最小持有期不得因分数下降卖出。
* hard exit / risk exit 可以突破 min_hold_days。

验收：

* 配置可加载。
* sell_score_threshold < candidate_min_trade_score。
* holding 配置进入 PortfolioConstraints 或独立 HoldingPolicy。
* 测试通过。

Task 2: 新增持仓状态和卖出决策模块

新增/修改：

* ml_stock_selector/portfolio/holding_policy.py
* ml_stock_selector/portfolio/constructor.py
* tests/test_holding_policy.py
* tests/test_portfolio_constructor_v2.py

新增数据结构：

@dataclass(frozen=True)
class HoldingState:
code: str
entry_date: str
entry_price: float
shares: float
holding_days: int
calendar_days: int
entry_trade_score: float | None
latest_trade_score: float | None
entry_reason: str | None

@dataclass(frozen=True)
class SellDecision:
code: str
should_sell: bool
reason: str
blocked: bool = False

实现函数：

def evaluate_sell_decision(
holding: HoldingState,
latest_row: pd.Series,
holding_policy: HoldingPolicy,
) -> SellDecision:
...

卖出规则：

1. hard_exit:

   * is_bse=true
   * is_st=true
   * data_quality_high_severity=true，如存在
   * 触发后应卖出，但如果 can_sell_next_open=false，则 blocked=true。

2. risk_exit:

   * risk_rank_pct >= risk_exit_rank_pct
   * 或 risk_prob >= risk_exit_prob
   * 可突破 min_hold_days。

3. score_exit:

   * holding_days >= min_hold_days
   * trade_score_v2 < sell_score_threshold
   * 且 can_sell_next_open=true 才能成交。

4. not_candidate_after_target_days:

   * holding_days >= target_hold_days
   * sell_if_not_candidate_after_target_days=true
   * 当前股票不在 candidate_pool
   * 则卖出。

5. max_hold_exit:

   * holding_days >= max_hold_days
   * force_exit_after_max_hold_days=true
   * 则卖出。

6. hold:

   * 未触发以上规则则继续持有。

验收：

* 未满 min_hold_days 时，单纯掉出 TopN 不卖。
* 未满 min_hold_days 但 risk_exit 触发，可卖。
* 满 target_hold_days 且不在 candidate_pool，可卖。
* 超过 max_hold_days，可卖。
* can_sell_next_open=false 时卖出 blocked，不应从持仓中移除。
* 测试通过。

Task 3: 重构组合构建为 current_holdings-aware

修改：

* ml_stock_selector/portfolio/constructor.py
* ml_stock_selector/portfolio/allocator.py
* tests/test_portfolio_constructor_v2.py
* tests/test_holding_aware_portfolio.py

新的组合构建流程必须为：

1. 输入 scored_candidates 和 current_holdings。
2. 先执行 hard filters，得到 tradable universe。
3. 构建 candidate_pool 和 core_pool。
4. 对 current_holdings 逐只执行 evaluate_sell_decision。
5. 保留未触发卖出的持仓 retained_holdings。
6. 对卖出 blocked 的股票继续保留，并记录 blocked_reason。
7. 计算 remaining_slots = hard_max_positions - retained_holdings_count。
8. 从 core_pool 中排除 retained_holdings 后补新仓。
9. core 不足时，从 candidate_pool 补新仓。
10. 新增股票数量 <= max_new_entries_per_day。
11. 如果 current_holdings 为空，则使用 max_initial_entries，而不是 max_new_entries_per_day。
12. 总持仓 <= hard_max_positions。
13. 如果候选不足，允许现金，不强行买满。

验收：

* target portfolio = retained_holdings + new_entries。
* 已有持仓未触发卖出时继续保留。
* 新增数受 max_new_entries_per_day 限制。
* 初始建仓可使用 max_initial_entries，不被限制为4只。
* 不再因为某股票当天没进 TopN 就立即卖出。
* entry_reason / hold_reason / exit_reason 可解释。
* 测试通过。

Task 4: 修改回测引擎，维护持仓生命周期

修改：

* ml_stock_selector/backtest/engine.py
* ml_stock_selector/backtest/execution.py
* tests/test_backtest_engine.py
* tests/test_holding_period.py

要求：

* 回测引擎必须维护 current_holdings，并在每日传入 construct_portfolio_targets_v2。
* current_holdings 中必须包含 entry_date、entry_price、holding_days、shares、entry_score 等字段。
* 每个交易日更新 holding_days。
* 卖出失败时继续持有，并在下一日继续评估。
* 买入成交后新增 entry_date 和 entry_price。
* 卖出成交后记录 exit_date、exit_price、holding_days、exit_reason。

扩展或新增输出字段：
ml_backtest_positions / orders 中记录：

* entry_date
* exit_date
* holding_days
* entry_trade_score
* exit_trade_score
* entry_reason
* exit_reason
* sell_blocked_reason

验收：

* 平均持仓时间不再长期接近 1 日。
* 持仓未满 min_hold_days 不会因为掉出目标池而卖出。
* can_sell_next_open=false 时卖出订单失败，持仓延续。
* max_new_entries_per_day 只影响新买入，不影响持仓延续。
* 测试通过。

Task 5: 增加组合构建和持仓周期诊断

修改：

* sql/create_ml_tables.sql
* ml_stock_selector/storage.py
* ml_stock_selector/portfolio/constructor.py
* ml_stock_selector/backtest/metrics.py
* ml_stock_selector/backtest/reports.py
* tests/test_portfolio_diagnostics.py
* tests/test_backtest_metrics.py

在 ml_portfolio_construction_diagnostics 中新增字段：

* retained_holdings_count
* sell_signal_count
* sell_executed_count
* sell_blocked_count
* hold_due_to_min_days_count
* hold_due_to_score_ok_count
* exit_due_to_score_count
* exit_due_to_risk_count
* exit_due_to_time_count
* exit_due_to_not_candidate_count
* avg_holding_days_current
* median_holding_days_current

在 ml_backtest_metrics 中新增：

* avg_holding_days
* median_holding_days
* max_holding_days
* holding_segment_count
* turnover_daily_avg
* sell_blocked_count
* hold_due_to_min_days_count
* exit_due_to_score_count
* exit_due_to_risk_count
* exit_due_to_time_count

验收：

* 回测报告能显示平均/中位持仓天数。
* 能解释卖出原因分布。
* 能解释为什么某股票被继续持有。
* 能看到 min_hold_days 是否实际生效。
* 测试通过。

Task 6: daily signal 使用同一套持仓感知逻辑

修改：

* ml_stock_selector/serving/daily_signal.py
* tests/test_daily_signal.py
* tests/test_holding_aware_portfolio.py

规则：

* daily signal 必须接收 current_holdings。
* 如果 current_holdings 为空，使用 max_initial_entries。
* 如果 current_holdings 非空，延续持仓按 holding_policy 判断。
* 输出中包含：

  * buy
  * sell
  * hold
  * sell_blocked
  * hold_reason
  * exit_reason
* daily signal 不应因为某持仓未进入今日 TopN 就直接卖出。

验收：

* daily signal 与 backtest 使用同一个 construct_portfolio_targets_v2。
* daily signal 输出可解释卖出/继续持有原因。
* 测试通过。

Task 7: 文档更新

修改：

* docs/ml_stock_selector_operating_notes.md
* README.md

补充说明：

1. horizon_d=5 不代表强制持有5天，但应给信号兑现时间。
2. 默认 holding policy:

   * min_hold_days=3
   * target_hold_days=5
   * max_hold_days=10
3. 买入阈值和卖出阈值不同，使用 hysteresis 降低换手。
4. core_pool / candidate_pool 负责买入候选。
5. sell decision 由 hard exit、risk exit、score exit、time exit 共同决定。
6. can_sell_next_open=false 时卖出不能假设成交。
7. max_new_entries_per_day 只限制新增，不限制持仓延续。

最终测试命令：
python -m pytest 
tests/test_holding_policy.py 
tests/test_holding_aware_portfolio.py 
tests/test_portfolio_constructor_v2.py 
tests/test_backtest_engine.py 
tests/test_holding_period.py 
tests/test_portfolio_diagnostics.py 
tests/test_backtest_metrics.py 
tests/test_daily_signal.py 
-v

手动验收 SQL：

1. 查看平均持仓时间：
   select
   avg_holding_days,
   median_holding_days,
   max_holding_days,
   holding_segment_count
   from ml_backtest_metrics
   where run_id = '<RUN_ID>'
   order by fold_id;

达标：

* avg_holding_days 应明显高于当前 1.56。
* median_holding_days 应明显高于 1。
* 如果 horizon_d=5，理想平均持仓大致应接近 3~6 个交易日，具体以回测效果为准。

2. 查看卖出原因：
   select
   exit_reason,
   count(*) as rows
   from ml_backtest_orders
   where run_id = '<RUN_ID>'
   and side = 'SELL'
   group by exit_reason
   order by rows desc;

达标：

* 不应全部是 dropped_from_target 或 rebalance_remove。
* 应能看到 score_exit、risk_exit、time_exit、not_candidate_after_target_days 等原因。

3. 查看 min_hold_days 是否生效：
   select
   sum(hold_due_to_min_days_count) as hold_due_to_min_days
   from ml_portfolio_construction_diagnostics
   where run_id = '<RUN_ID>';

达标：

* hold_due_to_min_days_count > 0，说明未满最小持有期的持仓被保留。

4. 查看 sell blocked：
   select
   sum(sell_blocked_count) as sell_blocked
   from ml_portfolio_construction_diagnostics
   where run_id = '<RUN_ID>';

达标：

* 如果存在跌停/停牌无法卖出情况，应有记录。
* sell_blocked 的股票不得从持仓中移除。

实现约束：

* 不要让行业、交易所、ST、停牌、可买卖字段进入模型特征。
* 不要取消 BSE / ST / paused / can_buy 的买入硬过滤。
* 不要把 min_hold_days 做成绝对不能卖；hard_exit 和 risk_exit 可以突破。
* 不要假设 can_sell_next_open=false 的股票卖出成功。
* 不要让 max_new_entries_per_day 限制总持仓。
* 不要强制每天买满；候选不足时允许现金。

提交要求：
每个 task 单独 commit。commit message 包含：
Constraint: holding-aware portfolio construction must align realized holding periods with model horizon and must not sell solely because a stock dropped out of daily TopN
Confidence: <high|medium|low>
Scope-risk: <narrow|moderate|broad>
Tested: <exact pytest command>
