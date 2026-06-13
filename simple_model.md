新增 VPA-ML 简化基准策略 abs_ranker_fixed_5d_risk_filter_v1：主模型预测 T+1 到 T+6 固定 5 日收益，risk model 预测未来 5 日大幅回撤概率；买入后默认固定持有 5 个交易日，到期卖出，只允许 risk_exit 提前退出。该策略用于替代当前复杂 score_exit / not_candidate / trailing / 多条件卖出逻辑，作为低过拟合 walk-forward 基准。

背景：
当前策略包含 score_exit、not_candidate_after_target_days、risk_exit、time_exit、trailing、core/candidate、动态阈值等复杂机制。回测显示不同年份对退出规则高度敏感，容易出现参数拟合。现在需要新增一个更干净的基准：

1. Absolute Ranker 负责主选股。
2. Risk Model 只做买入过滤和可选提前退出。
3. 卖出规则极简：固定持有 5 个交易日，到期卖出。
4. 不使用 score_exit。
5. 不使用 not_candidate_after_target_days。
6. 不使用 trailing_profit_exit。
7. 不因为股票掉出 TopN / candidate_pool 而卖出。

核心策略定义：

* T 日收盘生成信号。
* T+1 开盘买入。
* 默认持有 5 个交易日。
* holding_days >= 5 后卖出。
* 如果 can_sell_next_open=false，则卖出失败，继续持有并下一日继续尝试。
* 持仓期间只允许 risk_exit 提前卖出。
* 买入必须满足可交易条件和 risk 过滤。
* 训练和预测继续使用 executable universe，不允许不可买样本进入生产训练。

请实现以下任务：

Task 1: 新增策略配置

修改：

* config/ml_default.toml
* config/ml_walkforward.toml
* ml_stock_selector/config.py
* ml_stock_selector/portfolio/constraints.py
* tests/test_ml_config.py

新增配置段：

[portfolio.fixed_5d_risk_filter]
enabled = true
strategy_id = "abs_ranker_fixed_5d_risk_filter_v1"

holding_days = 5
target_positions = 10
hard_max_positions = 12
max_initial_entries = 10
max_new_entries_per_day = 12

min_abs_rank_pct = 0.70
risk_entry_max_rank_pct = 0.55
risk_exit_rank_pct = 0.85

min_adv20_amount = 50000000
exclude_bse = true
exclude_st = true
exclude_paused = true
require_can_buy_next_open = true
allow_cash = true

position_weight_mode = "equal_weight"
min_position_weight = 0.06
max_position_weight = 0.12

enable_risk_exit = true
enable_score_exit = false
enable_not_candidate_exit = false
enable_trailing_exit = false
enable_time_exit = true

规则：

* fixed_5d_risk_filter 是独立 strategy profile，不要破坏现有 portfolio.v2。
* enable_score_exit=false 时，任何分数下降都不得触发卖出。
* enable_not_candidate_exit=false 时，股票掉出 candidate_pool / TopN 不得触发卖出。
* enable_trailing_exit=false 时，不启用浮盈回撤卖出。
* enable_time_exit=true 时，仅 holding_days >= 5 触发到期卖出。
* max_new_entries_per_day 在该基准中可以等于 hard_max_positions，避免初始建仓/补仓被人为限制。

验收：

* 配置可加载。
* strategy_id 正确。
* 默认禁用 score_exit / not_candidate / trailing。
* 测试通过。

Task 2: 确认主模型标签和训练口径

修改：

* ml_stock_selector/sample_builder.py
* ml_stock_selector/label_builder.py 如需要
* tests/test_sample_builder.py
* tests/test_fixed_5d_labels.py

目标：
Absolute Ranker 主模型训练标签应对应：
T 日特征，T+1 开盘买入，T+6 收盘/可定义退出价计算未来 5 个交易日收益。

要求：

* 使用 label_base = "from_next_open"。
* 使用 horizon_d = 5。
* 主标签使用未来固定 5 日最终收益排序，例如 rank_label_abs 或等价字段。
* 训练样本必须 executable_only：

  * can_buy_next_open = true
  * is_bse = false
  * is_st = false
  * is_paused = false
  * adv20_amount >= min_adv20_amount
* can_buy_next_open / next_open / next_limit_up 不得进入 feature matrix，只能用于样本过滤和回测执行。
* risk model 使用同一 executable universe。
* risk label 保持未来 5 日大幅回撤风险，例如 future_max_drawdown_5d <= -5% 或当前已有 risk_label。

验收：

* Absolute Ranker 训练样本是 T 日预测 T+1~T+6 固定窗口收益。
* 不可买样本不进入训练。
* 风险模型训练样本 universe 与主模型一致。
* 行业、交易、可买字段不进入特征。
* 测试通过。

Task 3: 新增固定 5 日持仓组合构建器

新增/修改：

* ml_stock_selector/portfolio/fixed_horizon.py
* ml_stock_selector/portfolio/constructor.py
* tests/test_fixed_horizon_portfolio.py

实现函数：

construct_fixed_5d_risk_filter_targets(
scored_candidates: pd.DataFrame,
current_holdings: list[HoldingState],
constraints: FixedHorizonRiskFilterConfig,
trade_date: str,
) -> PortfolioTargetResult

买入逻辑：

1. 合并当日 predictions + tradeability metadata。
2. 执行买入硬过滤：

   * is_bse=false
   * is_st=false
   * is_paused=false
   * can_buy_next_open=true
   * adv20_amount >= min_adv20_amount
3. 执行 risk 买入过滤：

   * risk_rank_pct <= risk_entry_max_rank_pct
4. 执行主模型分数过滤：

   * absolute_rank_pct >= min_abs_rank_pct
5. 按 absolute_rank_pct 从高到低排序。
6. 排除已经持有且未到期的股票。
7. 选入直到 target_positions / hard_max_positions。
8. 使用 equal_weight，权重限制在 min_position_weight ~ max_position_weight。
9. 如果候选不足，允许现金。

持仓保留逻辑：

* current_holdings 中未触发 risk_exit 且 holding_days < holding_days 的股票继续持有。
* 持仓期间不因为 absolute_rank_pct 下降卖出。
* 持仓期间不因为不在 candidate_pool / TopN 卖出。

卖出逻辑：

* holding_days >= 5 -> time_exit。
* enable_risk_exit=true 且 risk_rank_pct >= risk_exit_rank_pct -> risk_exit。
* 如果 can_sell_next_open=false，则 sell_blocked，继续持有。
* 不允许 score_exit。
* 不允许 not_candidate_exit。
* 不允许 trailing_exit。

验收：

* 买入后未满 5 个交易日不会因为掉出 TopN 卖出。
* holding_days >= 5 才触发 time_exit。
* risk_exit 可以提前退出。
* score_exit / not_candidate_exit / trailing_exit 永远不触发。
* can_sell_next_open=false 时卖出失败且继续持有。
* 测试通过。

Task 4: 回测引擎支持 fixed_5d_risk_filter 策略

修改：

* scripts/run_ml_backtest.py
* ml_stock_selector/backtest/engine.py
* ml_stock_selector/backtest/execution.py
* tests/test_fixed_horizon_backtest.py
* tests/test_backtest_engine.py

要求：

* run_ml_backtest.py 支持参数：
  --strategy-id abs_ranker_fixed_5d_risk_filter_v1
* 当 strategy_id 为该值时，调用 construct_fixed_5d_risk_filter_targets。
* 每日回测必须维护 current_holdings。
* current_holdings 必须包含：

  * code
  * entry_date
  * entry_price
  * shares
  * holding_days
  * entry_abs_rank_pct
  * entry_risk_rank_pct
  * entry_reason
* 每日更新 holding_days。
* 到期卖出失败时继续持有。
* risk_exit 卖出失败时继续持有。
* 买入成交后记录 entry_date / entry_price。
* 卖出成交后记录 exit_date / exit_price / holding_days / exit_reason。

验收：

* 回测可以跑 fixed_5d 策略。
* 平均持仓天数应接近 5 个交易日，除非 risk_exit 或 sell_blocked 改变。
* 不再出现 score_exit / not_candidate_exit。
* 订单表中 exit_reason 只应包含 time_exit、risk_exit、hard_exit、sell_blocked 相关原因。
* 测试通过。

Task 5: 增加对照实验模式：无 risk_exit 版本

修改：

* config/ml_default.toml
* scripts/run_ml_backtest.py
* tests/test_fixed_horizon_backtest.py

目标：
增加一个可配置对照，用于验证 risk_exit 是否有效。

配置：
[portfolio.fixed_5d_no_risk_exit]
enabled = true
strategy_id = "abs_ranker_fixed_5d_no_risk_exit_v1"
holding_days = 5
enable_risk_exit = false
其他参数与 fixed_5d_risk_filter 相同。

规则：

* 买入时仍可使用 risk_entry_max_rank_pct 作为过滤，或允许配置关闭。
* 持仓期间不允许 risk_exit。
* 只在 holding_days >= 5 时卖出。

验收：

* 能跑两个策略：

  1. abs_ranker_fixed_5d_risk_filter_v1
  2. abs_ranker_fixed_5d_no_risk_exit_v1
* 可以比较 risk_exit 是否改善收益/回撤。
* 测试通过。

Task 6: 诊断与指标

修改：

* ml_stock_selector/backtest/metrics.py
* ml_stock_selector/backtest/reports.py
* sql/create_ml_tables.sql 如需要
* tests/test_backtest_metrics.py
* tests/test_fixed_horizon_reports.py

新增/确认指标：

* avg_holding_days
* median_holding_days
* max_holding_days
* holding_segment_count
* avg_positions
* avg_cash_ratio
* buy_count
* sell_count
* risk_exit_count
* time_exit_count
* sell_blocked_count
* avg_entry_abs_rank_pct
* avg_entry_risk_rank_pct
* realized_ret_by_exit_reason

报告要求：

* 输出每个 fold 的收益、年化、最大回撤、Calmar、胜率、平均持仓、平均现金、平均持有天数。
* 输出卖出原因盈亏统计。
* 输出 risk_exit 筛选前后候选数量。
* 输出 fixed_5d_risk_filter 与 fixed_5d_no_risk_exit 的对照结果。

验收：

* 报告能清楚判断：

  1. 主模型固定 5 日持有是否有效；
  2. risk_entry_filter 是否有效；
  3. risk_exit 是否有效；
  4. 平均持仓是否符合 5 日策略预期。
* 测试通过。

Task 7: 文档更新

修改：

* docs/ml_stock_selector_operating_notes.md
* README.md

补充说明：

* abs_ranker_fixed_5d_risk_filter_v1 是低过拟合基准策略。
* 策略目标是验证：

  * Absolute Ranker 能否预测 T+1 到 T+6 固定收益；
  * Risk Model 能否过滤大幅下跌风险。
* 该策略不使用：

  * score_exit
  * not_candidate_after_target_days
  * trailing_profit_exit
  * 动态卖出分数
* 默认卖出：

  * 持有 5 个交易日到期卖出；
  * risk_exit 可提前卖；
  * 卖不掉则继续持有。

最终测试命令：
python -m pytest 
tests/test_ml_config.py 
tests/test_sample_builder.py 
tests/test_fixed_5d_labels.py 
tests/test_fixed_horizon_portfolio.py 
tests/test_fixed_horizon_backtest.py 
tests/test_fixed_horizon_reports.py 
tests/test_backtest_engine.py 
tests/test_backtest_metrics.py 
-v

建议运行命令示例：

python scripts/run_ml_backtest.py 
--config config/ml_walkforward.toml 
--ml-db outputs/ml/ml.duckdb 
--run-id <RUN_ID> 
--strategy-id abs_ranker_fixed_5d_risk_filter_v1 
--score-version <SCORE_VERSION> 
--feature-set-id vpa_d_sequence 
--horizon-d 5 
--label-base from_next_open

对照版本：

python scripts/run_ml_backtest.py 
--config config/ml_walkforward.toml 
--ml-db outputs/ml/ml.duckdb 
--run-id <RUN_ID> 
--strategy-id abs_ranker_fixed_5d_no_risk_exit_v1 
--score-version <SCORE_VERSION> 
--feature-set-id vpa_d_sequence 
--horizon-d 5 
--label-base from_next_open

手动验收 SQL：

1. 检查卖出原因：
   select
   exit_reason,
   count(*) as rows,
   sum(realized_pnl) as pnl
   from ml_backtest_orders
   where run_id = '<RUN_ID>'
   and strategy_id = 'abs_ranker_fixed_5d_risk_filter_v1'
   and side = 'SELL'
   group by exit_reason
   order by rows desc;

达标：

* 不应出现 score_exit。
* 不应出现 not_candidate。
* 不应出现 trailing_exit。
* 应主要是 time_exit 和 risk_exit。

2. 检查平均持仓时间：
   select
   avg_holding_days,
   median_holding_days,
   max_holding_days,
   holding_segment_count
   from ml_backtest_metrics
   where run_id = '<RUN_ID>'
   and strategy_id = 'abs_ranker_fixed_5d_risk_filter_v1';

达标：

* avg_holding_days 应接近 5。
* 如果 risk_exit 较多，avg_holding_days 可以略低。
* 如果 sell_blocked 较多，avg_holding_days 可以略高。

3. 检查持仓数和现金：
   select
   fold_id,
   avg_positions,
   avg_cash_ratio,
   buy_count,
   sell_count
   from ml_backtest_metrics
   where run_id = '<RUN_ID>'
   and strategy_id = 'abs_ranker_fixed_5d_risk_filter_v1'
   order by fold_id;

达标：

* 平均持仓数与 target_positions 接近，除非候选不足或 risk 过滤过强。
* 不应因为卖出规则复杂导致异常高换手。

实现约束：

* 不要破坏现有 portfolio.v2 复杂策略。
* fixed_5d 策略必须是独立 profile。
* 不允许 can_buy_next_open / next_open / next_limit_up 进入模型特征。
* 不允许行业字段进入模型特征。
* 不允许 BSE / ST / paused / 不可买样本进入生产训练。
* fixed_5d 策略中不允许 score_exit / not_candidate_exit / trailing_exit。
* 回测必须遵守 can_buy_next_open 和 can_sell_next_open。

提交要求：
每个 task 单独 commit。commit message 包含：
Constraint: fixed 5-day baseline must align model label with execution horizon and avoid complex exit-rule overfitting
Confidence: <high|medium|low>
Scope-risk: <narrow|moderate|broad>
Tested: <exact pytest command>
