修正 VPA-ML portfolio v2 组合构建逻辑，解决 wf_2020 中目标股票过少的问题。

背景：
当前 wf_2020 诊断结果显示：

1. candidate 并不少，平均每天约 186.6 只 candidate，231/243 天满足最低 candidate 要求。
2. 真正卡在 core 层，全年只有 65 天有 core 股票，core 总行数 90，最终目标 87 行。
3. trade_score_v2 >= 0.80 有 2800 行，但硬过滤后只剩 96 行，主要原因是 low_adv、cannot_buy_next_open、ST，且有重叠。
4. max_new_entries_per_day=4 被错误用于空目标/初始建仓场景，导致每日最大持仓也被限制成 4 只。
5. core_pool 当前实际效果接近“三模型强信号交集”，但它应该是优先买入池，不应该是唯一买入池。

目标：

1. core_pool 作为强信号优先池。
2. candidate_pool 作为 core 不足时的补足池。
3. hard filters 应在 core/candidate 构建前统一执行。
4. max_new_entries_per_day 只限制“新增持仓”，不能限制初始建仓或总持仓。
5. 增加 max_initial_entries。
6. 增加 portfolio construction diagnostics，记录每天 core/candidate/过滤/最终入选数量。
7. 回测和 daily signal 都使用同一套 portfolio v2 逻辑。
8. 北交所、ST、停牌、不可买、低 ADV 仍然必须过滤。

请实现以下任务：

Task 1: 调整 PortfolioConstraints

修改：

* ml_stock_selector/portfolio/constraints.py
* config/ml_default.toml
* config/ml_walkforward.toml
* tests/test_portfolio_constructor_v2.py

新增配置：
[portfolio.v2]
target_positions = 12
hard_max_positions = 15
max_initial_entries = 12
max_new_entries_per_day = 4
max_industry_names = 3
max_unknown_industry_names = 1
min_trade_score = 0.65
core_min_trade_score = 0.75
candidate_min_trade_score = 0.65
min_candidate_pool_size = 5
allow_cash = true
min_adv20_amount = 50000000
exclude_bse = true

规则：

* max_initial_entries 只在 current_holdings 为空或初始建仓时使用。
* max_new_entries_per_day 只限制不在 current_holdings 中的新股票。
* hard_max_positions 永远是最终组合持仓上限。
* target_positions 是期望持仓数，但候选不足时允许少买和保留现金。

验收：

* current_holdings 为空时，最多可选 max_initial_entries，而不是 max_new_entries_per_day。
* current_holdings 非空时，已有持仓延续不计入 new entry。
* 新买入数量 <= max_new_entries_per_day。
* 最终持仓数量 <= hard_max_positions。

Task 2: 重构 core_pool / candidate_pool 逻辑

修改：

* ml_stock_selector/portfolio/constructor.py
* tests/test_portfolio_constructor_v2.py

流程必须改为：

1. 输入 scored_candidates。
2. 先执行 hard filters：

   * is_bse = false
   * is_st = false
   * is_paused = false
   * can_buy_next_open = true
   * adv20_amount >= min_adv20_amount
   * trade_score_v2 >= candidate_min_trade_score
3. 在 hard-filtered universe 中构建 core_pool。
4. 在 hard-filtered universe 中构建 candidate_pool。
5. 先从 core_pool 按 trade_score_v2 选。
6. core 不足 target_positions 时，从 candidate_pool 排除已选股票后补足。
7. 如果 candidate 也不足，则允许保留现金，不强行买满。

默认 core_pool 条件：

* absolute_rank_pct >= 0.75
* active_rank_pct >= 0.65
* risk_rank_pct <= 0.55
* trade_score_v2 >= core_min_trade_score

默认 candidate_pool 条件：

* (absolute_rank_pct >= 0.70 OR active_rank_pct >= 0.70)
* risk_rank_pct <= 0.65
* trade_score_v2 >= candidate_min_trade_score

验收：

* core_pool 优先入选。
* core_pool 不足时 candidate_pool 会补足。
* candidate_pool 不是只用于诊断，必须实际参与选股。
* 没有 core 但 candidate 足够时，仍可产生目标持仓。
* candidate 不足时允许少买。
* entry_reason 能区分 core_pool 和 candidate_pool。

Task 3: 修复 max_new_entries_per_day 语义

修改：

* ml_stock_selector/portfolio/constructor.py
* tests/test_portfolio_constructor_v2.py
* tests/test_backtest_engine.py

规则：

* 如果 current_holdings 为空：使用 max_initial_entries 控制初始建仓数量。
* 如果 current_holdings 非空：

  * 已有持仓可延续，不计入新买入数量。
  * 新股票数量 <= max_new_entries_per_day。
* 不允许 max_new_entries_per_day 把每日总持仓错误限制成 4 只。
* 如果已有持仓质量下降，可通过 trade_score/risk/可交易性规则卖出，但卖出不受 max_new_entries_per_day 限制。

验收：

* 空仓初始建仓时可选 10~12 只，不被限制为 4 只。
* 已有 10 只持仓时，最多新增 4 只。
* final_selected_count 可大于 max_new_entries_per_day。
* hard_max_positions 仍然生效。

Task 4: 增加组合构建诊断表

修改：

* sql/create_ml_tables.sql
* ml_stock_selector/storage.py
* ml_stock_selector/portfolio/constructor.py
* ml_stock_selector/backtest/engine.py
* tests/test_portfolio_diagnostics.py

新增表：
ml_portfolio_construction_diagnostics

字段至少包括：

* trade_date
* run_id
* fold_id
* portfolio_id
* score_version
* raw_candidate_count
* hard_filter_pass_count
* core_pool_size
* candidate_pool_size
* selected_from_core
* selected_from_candidate
* final_selected_count
* low_adv_rejected_count
* cannot_buy_rejected_count
* st_rejected_count
* paused_rejected_count
* bse_rejected_count
* low_trade_score_rejected_count
* high_risk_rejected_count
* industry_limit_blocked_count
* unknown_industry_limit_blocked_count
* max_new_entries_blocked_count
* cash_weight
* created_at

验收：

* 每个回测交易日写入一条 diagnostics。
* 能解释为什么目标股票少。
* wf_2020 中可以看到 core_pool_size、candidate_pool_size、selected_from_core、selected_from_candidate。
* 过滤原因计数支持重叠统计或明确说明是否 mutually exclusive。
* 测试通过。

Task 5: 回测和 daily signal 统一使用 portfolio v2

修改：

* scripts/run_ml_backtest.py
* ml_stock_selector/backtest/engine.py
* ml_stock_selector/serving/daily_signal.py
* tests/test_backtest_engine.py
* tests/test_daily_signal.py

规则：

* 回测和 daily signal 都调用同一个 construct_portfolio_targets_v2。
* 不允许回测中硬编码 PortfolioConstraints(min_trade_score=-999.0)。
* 必须从 config 读取 portfolio.v2。
* 回测必须传入 current_holdings，使 max_new_entries_per_day 正确生效。
* daily signal 也必须支持 current_holdings，否则初始建仓使用 max_initial_entries。

验收：

* run_ml_backtest.py 不再硬编码低阈值。
* 回测每日目标持仓可以延续已有持仓。
* daily signal 和 backtest 的 portfolio rules 一致。
* 测试通过。

Task 6: 增加诊断 SQL / 报告输出

修改：

* ml_stock_selector/backtest/reports.py
* docs/ml_stock_selector_operating_notes.md
* tests/test_backtest_reports.py

报告中增加：

* 平均 raw_candidate_count
* 平均 hard_filter_pass_count
* 平均 core_pool_size
* 平均 candidate_pool_size
* 平均 selected_from_core
* 平均 selected_from_candidate
* low_adv_rejected_count
* cannot_buy_rejected_count
* st_rejected_count
* max_new_entries_blocked_count
* empty_day_ratio
* avg_selected_count

验收：

* 回测报告能解释目标股票少是 core 不足、硬过滤过强、candidate 补足未生效，还是 max_new_entries 限制。
* 报告输出 wf_2020 每日 selected_count 分布。
* 测试通过。

最终测试命令：
python -m pytest 
tests/test_portfolio_constructor_v2.py 
tests/test_portfolio_diagnostics.py 
tests/test_backtest_engine.py 
tests/test_daily_signal.py 
tests/test_backtest_reports.py 
-v

手动验收 SQL：

1. 检查 candidate 是否补足：
   select
   avg(core_pool_size) as avg_core,
   avg(candidate_pool_size) as avg_candidate,
   avg(selected_from_core) as avg_selected_core,
   avg(selected_from_candidate) as avg_selected_candidate,
   avg(final_selected_count) as avg_selected
   from ml_portfolio_construction_diagnostics
   where run_id = '<RUN_ID>';

达标：

* selected_from_candidate 应明显大于 0。
* final_selected_count 不应长期等于 core_pool_size。
* candidate 足够时 final_selected_count 应接近 target_positions 或受行业/换手约束限制。

2. 检查 max_new_entries 是否不再限制总持仓：
   select
   trade_date,
   final_selected_count,
   max_new_entries_blocked_count
   from ml_portfolio_construction_diagnostics
   where run_id = '<RUN_ID>'
   order by trade_date;

达标：

* 空仓初始建仓日 final_selected_count 可以大于 4。
* 非初始建仓日新增数量受 max_new_entries_per_day 限制，但总持仓不被限制为 4。

3. 检查 entry_reason：
   select
   entry_reason,
   count(*) as rows
   from ml_portfolio_targets_daily
   where run_id = '<RUN_ID>'
   group by entry_reason
   order by rows desc;

达标：

* 应同时存在 core_pool 和 candidate_pool。
* 不应几乎全部来自 core_pool。

4. 检查北交所仍被剔除：
   select count(*) as bse_targets
   from ml_portfolio_targets_daily pt
   join ml_tradeability_daily t
   on pt.trade_date = t.trade_date
   and pt.code = t.code
   where pt.run_id = '<RUN_ID>'
   and (coalesce(t.is_bse, false) = true or pt.code like '920%');

达标：

* bse_targets = 0。

实现约束：

* 不要让 industry_code / industry_name / is_bse / is_st / can_buy_next_open 进入模型特征。
* 不要放松 BSE、ST、paused、cannot_buy_next_open 的硬过滤。
* core_pool 是优先强信号层，不是唯一买入层。
* candidate_pool 必须实际参与补足。
* max_new_entries_per_day 只限制新增，不限制总持仓。
* 候选不足时允许保留现金。

提交要求：
每个 task 单独 commit。commit message 包含：
Constraint: core_pool is priority-only, candidate_pool supplements positions, and max_new_entries limits only new entries
Confidence: <high|medium|low>
Scope-risk: <narrow|moderate|broad>
Tested: <exact pytest command>
