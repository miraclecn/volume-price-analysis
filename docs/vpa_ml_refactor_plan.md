# VPA-ML 系统重构改进计划

> 目标：将 `volume-price-analysis` 从“脚本 + DuckDB 表 + 零散 artifact + CLI/LLM 管理”的状态，重构为一个可复现、可审计、可展示、可接入实盘的机器学习量化研究与交易控制系统。

---

## 0. 总体原则

### 0.1 系统边界

```text
alpha-data / Market Loom：
只负责数据底座、PIT、质量审计、research_source.duckdb、audit JSON、本地数据健康 dashboard

volume-price-analysis：
负责 VPA 特征、ML feature mart、label、walk-forward、模型 artifact、回测、组合、信号、实盘监控、研究 dashboard
```

### 0.2 重构优先级

```text
第一优先级：固化 run_id / fold_id / artifact / config / metrics
第二优先级：保证不同实验结果不会互相覆盖
第三优先级：让模型参数与配置严格一致
第四优先级：拆分研究训练与生产激活
第五优先级：新增组合层和风控层
第六优先级：做浏览器 dashboard
```

不要先做 UI。否则只是把当前混乱状态可视化。

---

# Phase 0：冻结当前系统，保留可回滚基线

## 0.1 打 tag / 保存当前基线

任务：

```text
[ ] 给当前 main 打 tag，例如 vpa-ml-pre-refactor-202606
[ ] 备份当前 outputs/ml/ml.duckdb
[ ] 备份当前 outputs/ml/artifacts
[ ] 导出当前所有关键 run 的回测结果
[ ] 记录当前最优配置：expanding + 空置版本
```

建议新增文件：

```text
docs/refactor_baseline_202606.md
```

内容记录：

```text
当前最佳模型版本
当前最佳 walk-forward 结果
当前配置文件
当前数据源路径
当前模型 artifact 路径
当前已知问题
```

验收标准：

```text
[ ] 任意重构失败时，可以回到当前可运行版本
[ ] 当前最优回测结果有离线备份
[ ] 当前主分支、配置、数据库、artifact 均有明确快照
```

---

# Phase 1：明确 alpha-data 与 VPA 的边界

## 1.1 alpha-data 只作为上游数据服务

alpha-data / Market Loom 保留：

```text
research_source.duckdb
market_data_quality.json
dashboard.html
/api/summary
```

VPA 不做：

```text
[ ] 不在 VPA UI 里做原始数据下载
[ ] 不在 VPA UI 里做 PIT reference staging
[ ] 不在 VPA UI 里修复行业、ST、停牌、涨跌停数据
[ ] 不在 VPA UI 里管理 raw.duckdb
```

## 1.2 VPA 只读取 alpha-data 输出

VPA 侧只保留一个 read-only 数据健康检查：

```text
Data Health Card
----------------
alpha_data_db path
latest_trade_date
stock_bar_normalized_daily row count
quality audit status
UNKNOWN industry ratio
limit_up / limit_down missing count
incomplete trading dates
```

验收标准：

```text
[ ] VPA 不 import alpha-data 内部模块
[ ] VPA 只通过 DuckDB / JSON 读取 alpha-data 输出
[ ] VPA dashboard 只显示数据健康摘要，不管理数据底座
```

---

# Phase 2：建立统一 run_id / fold_id / experiment 管理

当前系统虽然部分表已经有 `run_id` / `fold_id` 字段，但它们没有贯穿所有结果表和 artifact，容易导致实验结果混淆。

## 2.1 新增 `ml_runs`

新增表：

```sql
create table if not exists ml_runs (
    run_id varchar primary key,
    run_type varchar not null,          -- walkforward / production_train / backtest / daily_signal / live
    experiment_name varchar,
    config_path varchar,
    config_hash varchar,
    git_commit varchar,
    alpha_data_db varchar,
    alpha_data_latest_date varchar,
    vpa_db varchar,
    ml_db varchar,
    feature_set_id varchar,
    feature_store_version varchar,
    label_version varchar,
    score_version varchar,
    artifact_root varchar,
    created_at varchar,
    started_at varchar,
    finished_at varchar,
    status varchar,                     -- created / running / success / failed
    notes varchar
);
```

## 2.2 新增 `ml_run_folds`

```sql
create table if not exists ml_run_folds (
    run_id varchar not null,
    fold_id varchar not null,
    train_start varchar,
    train_end varchar,
    valid_start varchar,
    valid_end varchar,
    test_start varchar,
    test_end varchar,
    gap_type varchar,                   -- one_year_gap / no_gap / rolling5_gap / rolling5_nogap
    embargo_days integer,
    status varchar,
    artifact_dir varchar,
    created_at varchar,
    primary key (run_id, fold_id)
);
```

## 2.3 新增 `RunContext`

新增代码：

```text
ml_stock_selector/runtime/run_context.py
```

职责：

```text
生成 run_id
读取 config
计算 config_hash
读取 git_commit
创建 artifact_root
写入 ml_runs
写入 ml_run_folds
提供所有后续模块共用的 run metadata
```

建议结构：

```python
@dataclass(frozen=True)
class RunContext:
    run_id: str
    run_type: str
    experiment_name: str
    config_path: Path
    config_hash: str
    git_commit: str
    artifact_root: Path
    feature_set_id: str
    label_version: str
    score_version: str
```

验收标准：

```text
[ ] 每次 walk-forward 都有唯一 run_id
[ ] 每个 fold 都有 run_id + fold_id
[ ] run_id 能追溯 config、git commit、数据源、artifact_root
[ ] 不允许无 run_id 的研究结果写入正式 ml_* 表
```

---

# Phase 3：修复所有结果表主键，防止实验互相覆盖

这是最高优先级之一。不同实验、不同模型、不同 fold 的结果必须能共存。

## 3.1 修改 `ml_backtest_nav`

建议字段：

```sql
run_id varchar not null,
fold_id varchar not null,
strategy_id varchar not null,
score_version varchar not null,
sim_date varchar not null,
nav double,
cash double,
gross_exposure double,
turnover double,
primary key (run_id, fold_id, strategy_id, score_version, sim_date)
```

## 3.2 修改 `ml_backtest_positions`

建议主键：

```sql
primary key (
    run_id,
    fold_id,
    strategy_id,
    score_version,
    sim_date,
    code
)
```

## 3.3 修改 `ml_backtest_orders`

建议字段和主键：

```sql
run_id varchar not null,
fold_id varchar not null,
strategy_id varchar not null,
score_version varchar not null,
sim_date varchar not null,
decision_date varchar not null,
code varchar not null,
side varchar not null,
order_seq integer not null,
qty double,
target_weight double,
order_px_ref varchar,
fill_px double,
status varchar,
reason varchar,
primary key (
    run_id,
    fold_id,
    strategy_id,
    score_version,
    sim_date,
    decision_date,
    code,
    side,
    order_seq
)
```

## 3.4 修改 `ml_portfolio_targets_daily`

建议主键：

```sql
primary key (
    trade_date,
    run_id,
    fold_id,
    portfolio_id,
    score_version,
    code
)
```

## 3.5 修改所有 upsert key

例如：

```python
upsert_dataframe(
    con,
    "ml_backtest_nav",
    result.nav,
    ["run_id", "fold_id", "strategy_id", "score_version", "sim_date"]
)
```

验收标准：

```text
[ ] 连续跑两个不同 run_id，结果不会互相覆盖
[ ] 同一个 fold 重跑同一个 run_id 可以正确覆盖自身
[ ] Dashboard 能同时展示多个 run 的 nav、orders、positions
```

---

# Phase 4：重构 artifact 结构

当前模型 artifact 信息偏薄，需要升级成完整 run artifact。

## 4.1 统一目录结构

建议目录：

```text
outputs/ml/runs/
  20260613_expanding_gap_vpa_d_sequence_h5/
    run_manifest.json
    config_snapshot.toml
    config_hash.txt
    git_commit.txt
    data_manifest.json
    feature_manifest.json
    label_manifest.json

    folds/
      wf_2020/
        fold_manifest.json

        models/
          absolute_ranker/
            model.pkl
            feature_schema.json
            params.json
            train_metrics.json
            feature_importance.csv

          active_ranker/
            model.pkl
            feature_schema.json
            params.json
            train_metrics.json
            feature_importance.csv

          risk_model/
            model.pkl
            feature_schema.json
            params.json
            train_metrics.json
            feature_importance.csv

        predictions/
          raw_predictions.parquet
          scored_predictions.parquet

        portfolio/
          targets.parquet
          diagnostics.parquet

        backtest/
          nav.parquet
          orders.parquet
          positions.parquet
          metrics.json
          yearly_metrics.json
          monthly_returns.parquet
          drawdown.parquet

    reports/
      walkforward_summary.html
      walkforward_summary.json
```

## 4.2 `run_manifest.json`

示例：

```json
{
  "run_id": "20260613_expanding_gap_v1",
  "run_type": "walkforward",
  "experiment_name": "expanding_gap",
  "created_at": "...",
  "git_commit": "...",
  "config_hash": "...",
  "config_path": "config/experiments/expanding_gap.toml",
  "alpha_data_db": "/home/nan/alpha-data-local/output/research_source.duckdb",
  "feature_set_id": "vpa_d_sequence",
  "label_base": "from_next_open",
  "horizon_d": 5,
  "score_version": "v2_three_model",
  "folds": ["wf_2020", "wf_2021", "wf_2022"]
}
```

## 4.3 `fold_manifest.json`

示例：

```json
{
  "run_id": "...",
  "fold_id": "wf_2021",
  "train_start": "2015-01-05",
  "train_end": "2019-12-31",
  "valid_start": "2020-01-01",
  "valid_end": "2020-12-31",
  "test_start": "2021-01-01",
  "test_end": "2021-12-31",
  "models": {
    "absolute": "...",
    "active": "...",
    "risk": "..."
  },
  "status": {
    "matrix_built": true,
    "models_trained": true,
    "predicted": true,
    "backtested": true
  }
}
```

验收标准：

```text
[ ] 任何 run 不依赖 LLM 记忆即可复现
[ ] 任何模型文件能追溯到 run_id/fold_id/config/data/git commit
[ ] 删除 DuckDB 明细后，仍能从 artifact 目录恢复主要报告
```

---

# Phase 5：让配置参数真正生效

当前配置文件中已经写了 LightGBM 参数，但训练代码中仍有硬编码参数。需要统一。

## 5.1 新增模型配置类

新增：

```text
ml_stock_selector/models/config.py
```

建议结构：

```python
@dataclass(frozen=True)
class LightGBMRankerConfig:
    objective: str
    metric: str
    n_estimators: int
    learning_rate: float
    num_leaves: int
    min_data_in_leaf: int
    feature_fraction: float
    bagging_fraction: float
    bagging_freq: int
    lambda_l2: float
    random_state: int
    eval_at: list[int]
```

风险模型：

```python
@dataclass(frozen=True)
class LightGBMRiskConfig:
    objective: str
    n_estimators: int
    learning_rate: float
    num_leaves: int
    min_data_in_leaf: int
    lambda_l2: float
    class_weight: str | None
    random_state: int
```

## 5.2 修改训练函数签名

当前：

```python
train_alpha_ranker(samples, feature_set_id, label_name, label_base, horizon_d, artifact_dir, deny_industry)
```

建议：

```python
train_alpha_ranker(
    samples,
    feature_set_id,
    label_name,
    label_base,
    horizon_d,
    artifact_dir,
    train_config: LightGBMRankerConfig,
    run_context: RunContext,
    fold_id: str,
    deny_industry: bool = False,
)
```

`risk_model` 同理。

## 5.3 params 写入 artifact 和 registry

输出：

```text
models/absolute_ranker/params.json
models/risk_model/params.json
```

同时写入：

```text
ml_model_registry.params_json
```

验收标准：

```text
[ ] config 中改 num_leaves，artifact params.json 会变化
[ ] registry.params_json 能看到真实训练参数
[ ] 相同 config_hash + data_hash + git_commit 能复现模型
[ ] 不再存在训练参数硬编码和配置不一致
```

---

# Phase 6：拆分研究训练和生产激活

当前训练完成后直接激活模型，不适合实盘系统。

## 6.1 新增 model bundle

新增表：

```sql
create table if not exists ml_model_bundles (
    bundle_id varchar primary key,
    run_id varchar not null,
    fold_id varchar,
    bundle_role varchar,              -- core / aggressive / production
    absolute_model_id varchar,
    active_model_id varchar,
    risk_model_id varchar,
    feature_set_id varchar,
    label_base varchar,
    horizon_d integer,
    score_version varchar,
    artifact_dir varchar,
    status varchar,                   -- candidate / approved / active / retired
    created_at varchar,
    activated_at varchar,
    deactivated_at varchar,
    notes varchar
);
```

## 6.2 训练不自动激活

新增：

```bash
python scripts/train_production_bundle.py \
  --config config/production/core_model.toml \
  --run-id prod_core_20260613
```

只做：

```text
[ ] train
[ ] register models
[ ] create candidate bundle
[ ] write artifact
```

## 6.3 单独激活

新增：

```bash
python scripts/activate_model_bundle.py \
  --bundle-id prod_core_20260613 \
  --confirm
```

激活时：

```text
[ ] 检查 bundle 三模型存在
[ ] 检查 feature schema 一致
[ ] 检查对应回测通过
[ ] 检查 production role 没有多个 active
[ ] 旧 bundle retired
[ ] 新 bundle active
```

验收标准：

```text
[ ] train 不再自动改变 production active model
[ ] daily signal 只读取 active bundle
[ ] 可以回滚到上一版 active bundle
```

---

# Phase 7：重构 walk-forward 实验入口

## 7.1 新增正式 CLI

新增：

```text
scripts/run_ml_walkforward.py
```

使用方式：

```bash
python scripts/run_ml_walkforward.py \
  --config config/experiments/expanding_gap.toml \
  --run-id 20260613_expanding_gap_v1 \
  --experiment-name expanding_gap \
  --force
```

支持：

```text
--fold-id wf_2021
--from-stage matrix
--to-stage backtest
--dry-run
--force
```

## 7.2 明确四类实验配置

建议配置目录：

```text
config/
  experiments/
    expanding_gap.toml
    expanding_nogap.toml
    rolling5_gap.toml
    rolling5_nogap.toml

  production/
    core_model.toml
    aggressive_model_1.toml

  portfolio/
    core_satellite_v1.toml

  live/
    qmt_default.toml
```

## 7.3 fold 生成器

新增：

```text
ml_stock_selector/split/fold_generator.py
```

支持：

```python
generate_expanding_folds(
    train_start="2015-01-05",
    first_test_year=2020,
    last_test_year=2026,
    gap_years=1,
)
```

支持实验类型：

```text
expanding_gap
expanding_nogap
rolling5_gap
rolling5_nogap
```

验收标准：

```text
[ ] 四种训练方式可以用统一入口跑
[ ] 每种方式生成的 fold 可打印、可审计、可入库
[ ] 不同实验结果不会互相覆盖
```

---

# Phase 8：补齐 metrics 与报告体系

## 8.1 每个 fold 输出标准指标

```text
annual_return
total_return
max_drawdown
calmar
sharpe
sortino
volatility
win_rate_daily
win_rate_monthly
turnover_daily_avg
cash_ratio_avg
position_count_avg
best_month
worst_month
max_consecutive_loss_days
max_consecutive_loss_months
```

## 8.2 walk-forward 汇总指标

```text
mean_annual_return
median_annual_return
min_annual_return
max_annual_return
std_annual_return
positive_year_ratio
max_of_max_drawdown
mean_drawdown
mean_calmar
worst_year
best_year
```

## 8.3 针对当前目标新增指标

当前目标：

```text
抓住 100%+ 年份
避免负收益和 30%+ 回撤
```

新增指标：

```text
negative_year_count
worst_year_return
drawdown_over_20_count
drawdown_over_30_count
high_return_capture_ratio
aggressive_year_capture_ratio
```

`high_return_capture_ratio` 定义：

```text
组合系统在激进模型高收益年份中捕捉到的收益比例
```

例如：

```text
激进模型某年 120%
组合系统某年 72%
capture = 72 / 120 = 60%
```

验收标准：

```text
[ ] dashboard 不只展示总年化，也展示最差年份和高收益捕捉率
[ ] 每个模型能明确归类为 core / aggressive / disabled
[ ] 选择模型不再只看平均收益
```

---

# Phase 9：新增策略组合层，而不是继续找单模型最优

目标：

```text
核心模型 + 激进模型 + 市场状态 + 模型健康 + 回撤控制
```

## 9.1 新增模块

```text
ml_stock_selector/strategy/
  regime.py
  model_health.py
  allocation.py
  ensemble.py
  risk_budget.py
```

## 9.2 市场状态表

```sql
create table if not exists ml_market_regime_daily (
    trade_date varchar primary key,
    trend_score double,
    breadth_score double,
    sentiment_score double,
    liquidity_score double,
    volatility_score double,
    final_regime varchar,       -- risk_on / neutral / risk_off / crash
    generated_at varchar
);
```

## 9.3 模型健康表

```sql
create table if not exists ml_model_health_daily (
    trade_date varchar not null,
    model_or_bundle_id varchar not null,
    rolling_20d_return double,
    rolling_60d_return double,
    rolling_20d_drawdown double,
    rolling_60d_drawdown double,
    equity_above_ma60 boolean,
    enabled_by_health boolean,
    reason varchar,
    primary key (trade_date, model_or_bundle_id)
);
```

## 9.4 策略分配表

```sql
create table if not exists ml_strategy_allocation_daily (
    trade_date varchar not null,
    strategy_id varchar not null,
    bundle_id varchar not null,
    role varchar,                  -- core / aggressive / cash
    raw_weight double,
    regime_multiplier double,
    health_multiplier double,
    drawdown_multiplier double,
    final_weight double,
    reason varchar,
    primary key (trade_date, strategy_id, bundle_id)
);
```

## 9.5 初始风险预算规则

市场状态分配：

```text
risk_on:
    core 60%
    aggressive 30%
    cash 10%

neutral:
    core 60%
    aggressive 10%
    cash 30%

risk_off:
    core 30%
    aggressive 0%
    cash 70%

crash:
    core 0%-10%
    aggressive 0%
    cash 90%-100%
```

账户回撤控制：

```text
回撤 < 5%     仓位系数 1.00
5%-10%        仓位系数 0.75
10%-15%       仓位系数 0.50
15%-20%       仓位系数 0.25
>20%          停止新开仓
```

验收标准：

```text
[ ] core 模型全年可运行
[ ] aggressive 模型只有 risk_on 且自身健康时启用
[ ] 组合层可以单独回测
[ ] 组合层结果和单模型结果可对比
```

---

# Phase 10：实盘信号与 QMT 链路标准化

## 10.1 新增 live tables

```sql
create table if not exists live_target_positions (
    trade_date varchar not null,
    account_id varchar not null,
    strategy_id varchar not null,
    code varchar not null,
    target_weight double,
    target_value double,
    source_bundle_id varchar,
    reason varchar,
    generated_at varchar,
    primary key (trade_date, account_id, strategy_id, code)
);
```

```sql
create table if not exists live_orders (
    order_id varchar primary key,
    trade_date varchar not null,
    account_id varchar not null,
    strategy_id varchar not null,
    code varchar not null,
    side varchar not null,
    order_qty double,
    order_price double,
    status varchar,
    block_reason varchar,
    created_at varchar,
    submitted_at varchar,
    updated_at varchar
);
```

```sql
create table if not exists live_fills (
    fill_id varchar primary key,
    order_id varchar,
    trade_date varchar,
    code varchar,
    side varchar,
    fill_qty double,
    fill_price double,
    fill_time varchar,
    commission double,
    tax double,
    slippage_bps double
);
```

```sql
create table if not exists live_risk_logs (
    trade_date varchar,
    account_id varchar,
    strategy_id varchar,
    check_name varchar,
    severity varchar,
    passed boolean,
    action varchar,
    reason varchar,
    created_at varchar
);
```

## 10.2 每日实盘流程

收盘后：

```text
[ ] 更新 alpha-data
[ ] 运行 VPA feature mart
[ ] 运行 daily prediction
[ ] 运行 strategy allocation
[ ] 生成 target positions
[ ] 风控检查
[ ] 导出 QMT order file
```

次日交易：

```text
[ ] QMT 读取订单
[ ] 委托
[ ] 成交回报
[ ] 写 live_orders / live_fills
[ ] 计算滑点和理论偏差
```

收盘后：

```text
[ ] 生成 live_pnl
[ ] 生成 model attribution
[ ] 生成风险日报
```

验收标准：

```text
[ ] 每笔实盘成交能追溯到模型、信号、target、订单
[ ] 能区分模型问题、风控问题、QMT 执行问题、滑点问题
[ ] 当日没有信号或没有订单时，也有明确状态记录
```

---

# Phase 11：浏览器 Dashboard，只读优先

技术建议：

```text
Streamlit + Plotly + DuckDB + Parquet
```

第一版只读，不触发训练，不触发下单。

## 11.1 目录结构

```text
dashboard/
  app.py
  pages/
    1_Run_Registry.py
    2_Walkforward_Compare.py
    3_Fold_Detail.py
    4_Model_Bundle.py
    5_Portfolio_Diagnostics.py
    6_Signal_Preview.py
    7_Live_Monitor.py
    8_Data_Health.py
```

## 11.2 Run Registry 页面

显示：

```text
run_id
experiment_name
run_type
status
feature_set_id
label_version
score_version
config_hash
git_commit
created_at
annual_return_mean
max_drawdown_worst
positive_year_ratio
```

## 11.3 Walk-forward Compare 页面

显示：

```text
各模型各 wf 年份收益
各模型各 wf 年份最大回撤
均值年化
最差年度
最大回撤
Calmar
高收益捕捉率
```

## 11.4 Fold Detail 页面

显示：

```text
权益曲线
回撤曲线
月度收益
持仓数量
换手率
订单列表
最大回撤区间
行业暴露
UNKNOWN 暴露
过滤原因
```

## 11.5 Model Bundle 页面

显示：

```text
absolute_model_id
active_model_id
risk_model_id
feature_schema_hash
params_json
train_metrics
artifact path
是否 active
是否 core/aggressive
```

## 11.6 Signal Preview 页面

显示：

```text
date
code
name
absolute_score
active_score
risk_prob
trade_score_v2
target_weight
signal_action
entry_reason
exit_reason
sell_blocked_reason
exclusion_reason
```

## 11.7 Live Monitor 页面

显示：

```text
今日目标持仓
今日订单
今日成交
拒单原因
成交率
滑点
理论收益 vs 实盘收益
QMT 状态
```

## 11.8 Data Health 页面

只显示 alpha-data 摘要，不管理 alpha-data：

```text
research_source.duckdb path
latest_trade_date
row count
data_quality JSON status
UNKNOWN industry ratio
limit missing count
incomplete trading dates
```

验收标准：

```text
[ ] dashboard 可以完全替代 CLI 查看结果
[ ] dashboard 不负责数据底座构建
[ ] dashboard 第一版只读，不触发训练/下单
```

---

# Phase 12：测试体系

## 12.1 Schema 测试

新增：

```text
tests/test_ml_schema_run_keys.py
```

检查：

```text
[ ] backtest_nav 主键包含 run_id/fold_id
[ ] positions 主键包含 run_id/fold_id
[ ] orders 主键包含 run_id/fold_id
[ ] targets 主键包含 run_id/fold_id/score_version
```

## 12.2 Artifact 测试

新增：

```text
tests/test_artifact_manifest.py
```

检查：

```text
[ ] 每个 run 有 run_manifest.json
[ ] 每个 fold 有 fold_manifest.json
[ ] 每个模型有 params.json / feature_schema.json / train_metrics.json
[ ] manifest 中路径真实存在
```

## 12.3 配置生效测试

新增：

```text
tests/test_model_config_applied.py
```

检查：

```text
[ ] config 里的 num_leaves 传入模型
[ ] params.json 与 config 一致
[ ] registry.params_json 与 artifact params.json 一致
```

## 12.4 实验隔离测试

新增：

```text
tests/test_run_isolation.py
```

流程：

```text
[ ] 跑 run_A
[ ] 跑 run_B
[ ] 确认 run_A nav 没被覆盖
[ ] 确认 run_B nav 独立存在
```

## 12.5 生产激活测试

新增：

```text
tests/test_model_bundle_activation.py
```

检查：

```text
[ ] train 不自动激活
[ ] activate_bundle 后只有一个 active production bundle
[ ] 可以回滚旧 bundle
```

---

# 推荐执行顺序

## 第 1 批：必须先做

```text
[ ] Phase 0：冻结当前系统
[ ] Phase 2：新增 ml_runs / ml_run_folds / RunContext
[ ] Phase 3：修复所有弱主键和 upsert key
[ ] Phase 5：让 LightGBM 参数真正从 config 生效
```

完成后，实验不会互相覆盖，也能确认“参数固定”是真的固定。

## 第 2 批：artifact 固化

```text
[ ] Phase 4：完整 run/fold/model artifact 目录
[ ] manifest 化 config / data / feature / label / model / metrics
[ ] 所有模型注册写入 run_id/fold_id/params_json/metrics_json
```

完成后，半年后仍然能复现某次模型。

## 第 3 批：生产安全

```text
[ ] Phase 6：拆分训练和激活
[ ] 新增 model_bundle
[ ] daily signal 改为读取 active bundle
```

完成后，研究训练不会误伤实盘模型。

## 第 4 批：组合系统

```text
[ ] Phase 8：补齐核心指标
[ ] Phase 9：新增 core/aggressive/regime/risk_budget 策略组合层
```

完成后，才能系统性解决“抓住 100%+ 年份，但避免负收益和 30%+ 回撤”。

## 第 5 批：界面和实盘

```text
[ ] Phase 11：只读 dashboard
[ ] Phase 10：live orders / fills / risk logs
[ ] dashboard 加 live monitor
```

完成后，再逐步减少 CLI 和 LLM 的参与。

---

# 最终目标结构

```text
alpha-data / Market Loom
    ↓
research_source.duckdb + audit JSON
    ↓
volume-price-analysis
    ├── vpa feature pipeline
    ├── ml feature mart
    ├── label builder
    ├── run manager
    ├── walk-forward engine
    ├── model artifact registry
    ├── backtest engine
    ├── strategy allocation layer
    ├── daily signal engine
    ├── live trading monitor
    └── dashboard
```

LLM 的角色变成：

```text
读取 run / metrics / artifact / live logs
→ 生成解释和复盘
```

而不是：

```text
替系统记住模型状态
替系统判断当前用了哪个版本
替系统管理实验结果
```

---

# 一句话总结

先把 `run_id`、`artifact`、`schema`、`config`、`metrics`、`backtest result` 固化，再做浏览器界面。

VPA UI 应该是研究与实盘控制台，不是数据底座控制台。
