# VPA-ML Three-Model v2 Closure Tasks

> **适用项目**：`miraclecn/volume-price-analysis`  
> **适用范围**：`ml_stock_selector` 子系统  
> **目标版本**：Three-Model v2 Closure  
> **用途**：可直接作为 Codex `/goal` 执行文档，也可作为人工实施验收清单。
>
> 本文档覆盖三类任务：
>
> 1. 修复当前仍不是完整三模型 v2 的 6 个部分；
> 2. 明确 2014 年作为 warm-up，2015 年样本纳入正式训练 / 回测；
> 3. 在 VPA-ML 阶段剔除北交所股票，不让其进入训练、预测、组合、回测、日信号。
>
> 当前已知基础状态：
>
> ```text
> ml_tradeability_daily：11,216,101 行，2015-01-05 ~ 2026-05-29
> ml_feature_mart_daily：11,216,101 行，2015-01-05 ~ 2026-05-29
> ml_labels_daily：49,354,556 行，2015-01-05 ~ 2026-05-28
> ```
>
> 这说明三模型所需的数据层基本完成；当前缺口主要在 **walk-forward 训练、fold 级预测、带交易元数据的组合回测、配置与审计层**。

---

## 0. 总体原则

### 0.1 目标

把当前 VPA-ML 从“可以 smoke 跑三模型”升级为“可严肃验证三模型策略”的完整闭环：

```text
ml_feature_mart_daily + ml_labels_daily
        ↓
fold-level 三模型训练
        ↓
fold-level batch prediction
        ↓
ml_predictions_daily
        ↓
prediction + tradeability metadata
        ↓
portfolio targets
        ↓
T+1 execution backtest
        ↓
fold metrics / run metrics
        ↓
production active 三模型
        ↓
daily signal
```

### 0.2 三模型定义

三模型 v2 包括：

```text
Absolute Ranker:
  学习价格/成交量/VPA 量价关系 → 未来绝对收益排名

Active Ranker:
  学习价格/成交量/VPA 量价关系 → 未来相对大盘/行业的主动收益排名

Risk Model:
  学习价格/成交量/VPA 量价关系 → 未来回撤风险概率
```

### 0.3 行业信息边界

行业信息 **不作为训练特征**。

允许用途：

```text
1. active label 的 benchmark 计算；
2. 组合行业分散约束；
3. UNKNOWN 行业限制；
4. 回测与日信号报告展示。
```

禁止用途：

```text
1. 不进入 features_json；
2. 不进入 feature_matrix；
3. 不作为 LightGBM 特征；
4. 不让模型直接学习行业身份。
```

### 0.4 北交所处理原则

北交所股票在 VPA-ML 阶段统一剔除：

```text
训练样本：剔除
批量预测：剔除，或标记 ineligible 且不写入策略预测
组合构建：剔除
回测：剔除
daily signal：剔除
```

`alpha-data` 和 VPA 可保留北交所数据；当前任务只要求 VPA-ML 策略域排除。

### 0.5 2015 样本原则

已确认：

```text
2014 = warm-up / lookback history
2015 = 正式可用训练样本
```

所以所有正式 walk-forward 配置应使用：

```text
train_start = 2015-01-05
```

不再默认从 2016 开始。

---

# Phase 1: Universe and Time Boundary Alignment

## Phase 1 目标

统一 VPA-ML 的训练宇宙和时间边界：

```text
CN A-shares excluding BSE
train_start = 2015-01-05
2014 only warm-up
```

---

## Task 1.1: Add Universe Configuration for Excluding BSE

### 目标

在配置层明确 VPA-ML 策略宇宙：沪深 A 股，不含北交所。

### 修改文件

```text
config/ml_default.toml
config/ml_walkforward.toml
config/ml_smoke.toml  # 如果存在
ml_stock_selector/config.py
ml_stock_selector/constants.py
tests/test_ml_config.py
tests/test_universe_filter.py
```

### 新增配置

```toml
[universe]
name = "cn_a_ex_bse"
exclude_bse = true
exclude_st = true
exclude_paused = true
```

### 新增常量

```python
UNIVERSE_CN_A_EX_BSE = "cn_a_ex_bse"
```

### 规则

1. `exclude_bse` 默认值应为 `true`。
2. 如果配置缺失 `[universe]`，系统应使用默认宇宙 `cn_a_ex_bse`。
3. `exclude_st` 和 `exclude_paused` 属于交易/组合硬过滤；`exclude_bse` 属于策略宇宙过滤，应在训练、预测、组合、回测、日信号全流程生效。

### 验收标准

- `load_ml_config()` 能读取 `[universe]`。
- 缺省配置时 `exclude_bse = true`。
- 配置测试通过。
- 不影响已有 smoke 配置加载。

### 测试命令

```bash
python -m pytest \
  tests/test_ml_config.py \
  tests/test_universe_filter.py \
  -v
```

---

## Task 1.2: Implement BSE Detection and Universe Filter

### 目标

实现统一的北交所识别和过滤函数，供训练、预测、回测、日信号复用。

### 修改文件

```text
ml_stock_selector/universe.py
tests/test_universe_filter.py
```

### 新增接口

```python
def detect_is_bse(
    frame: pd.DataFrame,
    code_col: str = "code",
    exchange_col: str | None = "exchange",
    board_col: str | None = "board",
) -> pd.Series:
    ...

def apply_universe_filter(
    frame: pd.DataFrame,
    exclude_bse: bool = True,
    code_col: str = "code",
) -> pd.DataFrame:
    ...
```

### 识别优先级

1. 如果存在 `is_bse` 字段，优先使用。
2. 如果存在 `exchange` 字段，识别：
   ```text
   BSE, BJ, 北京证券交易所
   ```
3. 如果存在 `board` 字段，识别：
   ```text
   北交所, Beijing Stock Exchange, BSE
   ```
4. fallback 使用代码前缀：
   ```text
   920
   ```
5. 历史 `43/83/87/88` 不建议默认全剔除，除非存在 exchange/board 佐证；避免误伤新三板或历史编码数据。

### 规则

- 默认只用 `920` 作为纯代码 fallback。
- 如果业务方确认历史北交所老代码映射表可用，再通过 `security metadata` 识别，而不是直接扩大前缀。
- 函数应返回过滤前后行数审计信息或至少支持调用方统计。

### 验收标准

- 有 `is_bse` 字段时按字段过滤。
- 有 `exchange=BSE/BJ` 时识别为北交所。
- `920xxx` fallback 识别为北交所。
- `830xxx` 不应仅凭前缀默认剔除。
- `apply_universe_filter(exclude_bse=True)` 后无北交所样本。
- 测试通过。

### 测试命令

```bash
python -m pytest tests/test_universe_filter.py -v
```

---

## Task 1.3: Update Walk-Forward Config to Include 2015

### 目标

将正式 walk-forward 配置改为从 2015-01-05 开始训练，并提供多 fold 滚动方案。

### 修改文件

```text
config/ml_walkforward.toml
config/ml_default.toml
ml_stock_selector/config.py
tests/test_ml_config.py
```

### 推荐配置

```toml
[split]
embargo_days = 10

folds = [
  { fold_id = "wf_2020", train_start = "2015-01-05", train_end = "2018-12-31", valid_start = "2019-01-01", valid_end = "2019-12-31", test_start = "2020-01-01", test_end = "2020-12-31" },

  { fold_id = "wf_2021", train_start = "2015-01-05", train_end = "2019-12-31", valid_start = "2020-01-01", valid_end = "2020-12-31", test_start = "2021-01-01", test_end = "2021-12-31" },

  { fold_id = "wf_2022", train_start = "2015-01-05", train_end = "2020-12-31", valid_start = "2021-01-01", valid_end = "2021-12-31", test_start = "2022-01-01", test_end = "2022-12-31" },

  { fold_id = "wf_2023", train_start = "2015-01-05", train_end = "2021-12-31", valid_start = "2022-01-01", valid_end = "2022-12-31", test_start = "2023-01-01", test_end = "2023-12-31" },

  { fold_id = "wf_2024", train_start = "2015-01-05", train_end = "2022-12-31", valid_start = "2023-01-01", valid_end = "2023-12-31", test_start = "2024-01-01", test_end = "2024-12-31" },

  { fold_id = "wf_2025", train_start = "2015-01-05", train_end = "2023-12-31", valid_start = "2024-01-01", valid_end = "2024-12-31", test_start = "2025-01-01", test_end = "2025-12-31" },

  { fold_id = "wf_2026_ytd", train_start = "2015-01-05", train_end = "2024-12-31", valid_start = "2025-01-01", valid_end = "2025-12-31", test_start = "2026-01-01", test_end = "2026-05-28" }
]
```

### 规则

1. `config/ml_default.toml` 可保留一个 smoke fold，但必须注明不是正式回测。
2. `config/ml_walkforward.toml` 必须包含多 fold。
3. 所有正式实验脚本默认推荐读取 `config/ml_walkforward.toml`。
4. `embargo_days` 必须参与训练/验证/测试边界处理。

### 验收标准

- 配置可加载。
- fold_id 唯一。
- 所有 fold 时间边界有序且不重叠。
- 每个 fold 的 `train_start = 2015-01-05`。
- `embargo_days = 10` 被解析。
- 测试通过。

### 测试命令

```bash
python -m pytest tests/test_ml_config.py -v
```

---

# Phase 2: Complete Three-Model Walk-Forward

## Phase 2 目标

修复当前 walk-forward 仍为旧单模型的问题，实现 fold 级三模型训练、注册、预测、回测和指标落库。

---

## Task 2.1: Refactor Walk-Forward Runner to Train Three Models per Fold

### 当前问题

当前：

```text
ml_stock_selector/backtest/walkforward.py
```

仍然只训练：

```text
alpha_ranker + rank_label
```

没有：

```text
Active Ranker
Risk Model
trade_score_v2
fold 级 model registry
```

### 目标

每个 fold 训练三模型：

```text
Absolute Ranker -> rank_label_abs
Active Ranker   -> rank_label_active
Risk Model      -> risk_label
```

### 修改文件

```text
ml_stock_selector/backtest/walkforward.py
ml_stock_selector/models/alpha_ranker.py
ml_stock_selector/models/active_ranker.py
ml_stock_selector/models/risk_model.py
ml_stock_selector/registry.py
ml_stock_selector/datasets.py
tests/test_walkforward.py
tests/test_active_ranker.py
tests/test_risk_model_probability.py
```

### 新增/调整接口

```python
@dataclass(frozen=True)
class WalkForwardFoldConfig:
    fold_id: str
    train_start: str
    train_end: str
    valid_start: str
    valid_end: str
    test_start: str
    test_end: str

@dataclass(frozen=True)
class ThreeModelFoldArtifacts:
    fold_id: str
    absolute_model_id: str
    active_model_id: str
    risk_model_id: str
    feature_schema_uri: str

@dataclass(frozen=True)
class WalkForwardFoldResult:
    fold_id: str
    artifacts: ThreeModelFoldArtifacts
    predictions: pd.DataFrame
    targets: pd.DataFrame
    metrics: dict[str, float]
```

### 训练规则

1. 每个 fold 只使用 fold 的 train/valid 数据训练。
2. test 区间只用于预测与回测，不参与训练。
3. 三个模型共享同一套 price-volume / VPA 特征 schema。
4. `industry_code / industry_name / industry_unknown / is_bse / exchange / board` 不得进入 feature matrix。
5. 每个 fold 的模型写入 `ml_model_registry`，但：
   ```text
   is_active = false
   ```
6. 只有最终生产模型才可激活。

### label 对应关系

```text
Absolute Ranker:
  label_name = rank_label_abs

Active Ranker:
  label_name = rank_label_active

Risk Model:
  label_name = risk_label
```

### 达标标准

- walk-forward 不再调用旧单模型 `rank_label` 路径作为唯一逻辑。
- 每个 fold 产生 3 个 model_id。
- fold 模型注册成功但不激活为 production active。
- test 区间预测只使用当前 fold 模型。
- 输出中包含 `fold_id / run_id / score_version`。
- 测试通过。

### 测试命令

```bash
python -m pytest \
  tests/test_walkforward.py \
  tests/test_active_ranker.py \
  tests/test_risk_model_probability.py \
  -v
```

---

## Task 2.2: Make Training Sample Builder Universe-Aware

### 目标

训练阶段剔除北交所，并使用 2015 起样本。

### 修改文件

```text
ml_stock_selector/sample_builder.py
ml_stock_selector/universe.py
tests/test_sample_builder.py
tests/test_universe_filter.py
tests/test_training_samples_exclude_bse.py
```

### 规则

训练样本构造流程应为：

```text
1. 读取 ml_feature_mart_daily
2. 读取 ml_labels_daily
3. 关联 horizon_d / label_base / feature_set_id
4. 关联 ml_tradeability_daily 或可用 metadata
5. 过滤 trade_date >= 2015-01-05
6. 过滤 is_bse = false
7. 过滤不可用于 ml_feature / ml_label 的质量问题
8. 输出训练样本
```

### 新增接口建议

```python
def build_training_samples(
    feature_mart: pd.DataFrame,
    labels: pd.DataFrame,
    feature_set_id: str,
    horizon_d: int,
    label_base: str,
    label_name: str,
    tradeability: pd.DataFrame | None = None,
    universe_config: UniverseConfig | None = None,
    start_date: str | None = "2015-01-05",
    end_date: str | None = None,
) -> pd.DataFrame:
    ...
```

### 达标标准

- 2015 年样本能进入训练样本。
- 北交所样本不进入训练样本。
- `industry_code` 可以作为 metadata 保留，但不得进入 features。
- 同一套样本构建函数可用于 `rank_label_abs / rank_label_active / risk_label`。
- 测试通过。

### 测试命令

```bash
python -m pytest \
  tests/test_sample_builder.py \
  tests/test_training_samples_exclude_bse.py \
  tests/test_universe_filter.py \
  -v
```

---

## Task 2.3: Ensure Feature Matrix Excludes Industry and Universe Metadata

### 目标

确认模型只学习价格、成交量、VPA 多周期量价特征，不学习行业或北交所身份。

### 修改文件

```text
ml_stock_selector/feature_matrix.py
tests/test_feature_matrix.py
tests/test_no_industry_training_features.py
```

### 必须排除字段

无论它们出现在 DataFrame metadata 还是 features_json，都不得进入模型矩阵：

```text
industry_code
industry_name
industry_unknown
industry_missing
exchange
board
is_bse
is_st
is_paused
can_buy_next_open
can_sell_next_open
```

说明：

- `is_st / is_paused / can_buy_next_open` 属于交易筛选信息，不是训练特征。
- 如果未来要把流动性类特征纳入模型，应通过明确的 feature allowlist，而不是自动读取 metadata。

### 推荐机制

使用 feature allowlist / denylist：

```python
FEATURE_METADATA_DENYLIST = {
    "industry_code",
    "industry_name",
    "industry_unknown",
    "industry_missing",
    "exchange",
    "board",
    "is_bse",
    "is_st",
    "is_paused",
    "can_buy_next_open",
    "can_sell_next_open",
}
```

### 达标标准

- `features_json` 中即使误含行业字段，feature_matrix 也会排除。
- 输出矩阵列名不包含行业/交易/宇宙字段。
- 训练日志输出 excluded metadata fields。
- 测试通过。

### 测试命令

```bash
python -m pytest \
  tests/test_feature_matrix.py \
  tests/test_no_industry_training_features.py \
  -v
```

---

# Phase 3: Complete Fold-Aware Batch Prediction

## Phase 3 目标

修复批量预测“是三模型但不完整”的问题：支持日期范围、fold/run/model 作用域，输出可审计预测。

---

## Task 3.1: Add Fold/Run/Date Scope to Batch Prediction

### 当前问题

`scripts/run_ml_batch_predict.py` 当前：

```text
能加载三模型并写 absolute/active/risk；
但读全量 mart；
没有日期范围；
没有 fold_id/run_id/model_id 作用域；
写出的预测混在一起。
```

### 修改文件

```text
scripts/run_ml_batch_predict.py
ml_stock_selector/prediction.py
ml_stock_selector/registry.py
tests/test_ml_prediction.py
tests/test_three_model_prediction_merge.py
```

### 新增 CLI 参数

```bash
python scripts/run_ml_batch_predict.py \
  --config config/ml_walkforward.toml \
  --start-date 2021-01-01 \
  --end-date 2021-12-31 \
  --run-id wf_three_model_v2_001 \
  --fold-id wf_2021 \
  --score-version v2_three_model \
  --absolute-model-id <model_id> \
  --active-model-id <model_id> \
  --risk-model-id <model_id>
```

### 规则

1. 必须支持 `start_date / end_date`。
2. 必须支持 `run_id / fold_id`。
3. 必须显式记录三个模型 ID。
4. 如果未提供 model_id，可以从 registry 查询 active 模型，但该模式只用于 production daily signal，不用于 walk-forward。
5. walk-forward 内部调用预测函数时必须传入 fold 模型 ID，不得读取 production active 模型。

### 输出字段要求

`ml_predictions_daily` 至少包含：

```text
trade_date
code
horizon_d
feature_set_id
score_version
run_id
fold_id

absolute_model_id
active_model_id
risk_model_id

absolute_score
absolute_rank_pct
active_score
active_rank_pct
risk_prob
risk_rank_pct

trade_score_v2
generated_at
```

如果当前 schema 不方便扩展，可新增：

```text
ml_prediction_components_daily
```

但最终 portfolio/backtest 必须能读到三模型融合结果。

### 达标标准

- 指定日期范围时只预测该范围。
- 同一日期可区分不同 run/fold/score_version。
- walk-forward 预测不使用 production active 模型。
- 预测结果中三模型 ID 完整。
- 测试通过。

### 测试命令

```bash
python -m pytest \
  tests/test_ml_prediction.py \
  tests/test_three_model_prediction_merge.py \
  -v
```

---

## Task 3.2: Apply Universe Filter During Batch Prediction

### 目标

批量预测阶段剔除北交所。

### 修改文件

```text
ml_stock_selector/prediction.py
scripts/run_ml_batch_predict.py
tests/test_prediction_exclude_bse.py
tests/test_universe_filter.py
```

### 规则

1. batch predict 读取 `ml_feature_mart_daily` 后，应关联 `ml_tradeability_daily` 或 metadata，以识别 `is_bse`。
2. `exclude_bse = true` 时，不对北交所股票生成策略预测。
3. 输出审计字段：
   ```text
   input_rows
   excluded_bse_rows
   predicted_rows
   ```
4. 如果保留北交所预测，则必须标记：
   ```text
   eligible_universe = false
   ```
   且不能进入组合；但第一版建议直接不写入预测。

### 达标标准

- `ml_predictions_daily` 中无北交所预测，或北交所预测全部 `eligible_universe=false`。
- daily / walk-forward 预测均遵守同一规则。
- 测试通过。

### 测试命令

```bash
python -m pytest \
  tests/test_prediction_exclude_bse.py \
  tests/test_universe_filter.py \
  -v
```

---

## Task 3.3: Compute and Persist trade_score_v2

### 目标

批量预测不应只落 raw prediction，还应落地最终可供组合使用的 `trade_score_v2`。

### 修改文件

```text
ml_stock_selector/scoring.py
ml_stock_selector/prediction.py
tests/test_ml_scoring.py
tests/test_three_model_prediction_merge.py
```

### 默认公式

第一版使用均衡权重：

```text
trade_score_v2 =
0.55 * absolute_rank_pct
+ 0.35 * active_rank_pct
- 0.25 * risk_rank_pct
```

说明：

- `absolute_rank_pct` 越高越好；
- `active_rank_pct` 越高越好；
- `risk_rank_pct` 越高越危险，是扣分项。

### 后续扩展

支持配置：

```toml
[scoring.v2]
abs_weight = 0.55
active_weight = 0.35
risk_weight = 0.25
min_trade_score = 0.80
score_version = "v2_three_model"
```

### 达标标准

- `ml_predictions_daily.trade_score_v2` 非空。
- 同一 trade_date 内 rank_pct 合理分布在 0~1。
- 风险越高，其他条件相同下 trade_score_v2 越低。
- 测试通过。

### 测试命令

```bash
python -m pytest \
  tests/test_ml_scoring.py \
  tests/test_three_model_prediction_merge.py \
  -v
```

---

# Phase 4: Complete Three-Model Backtest and Portfolio Construction

## Phase 4 目标

修复当前批量回测半旧半新的问题：回测必须读取预测 + 交易元数据 + 配置阈值，并产出 fold/run 级指标。

---

## Task 4.1: Build Backtest Candidate Frame with Tradeability Metadata

### 当前问题

`scripts/run_ml_backtest.py` 当前只读：

```text
ml_predictions_daily
```

缺少：

```text
industry_code
is_st
is_paused
can_buy_next_open
adv20_amount
```

导致 ST、停牌、可买、行业约束、流动性约束不能正确生效。

### 修改文件

```text
scripts/run_ml_backtest.py
ml_stock_selector/backtest/engine.py
ml_stock_selector/backtest/data_access.py
tests/test_backtest_candidate_join.py
tests/test_backtest_execution.py
```

### 新增接口

```python
def load_backtest_candidates(
    con: duckdb.DuckDBPyConnection,
    start_date: str,
    end_date: str,
    run_id: str,
    fold_id: str | None,
    score_version: str,
    feature_set_id: str,
    horizon_d: int,
) -> pd.DataFrame:
    ...
```

### join 逻辑

```sql
SELECT
    p.*,
    t.industry_code,
    t.industry_name,
    t.is_st,
    t.is_paused,
    t.adv20_amount,
    t.can_buy_next_open,
    t.can_sell_next_open,
    t.next_open,
    t.next_limit_up,
    t.next_limit_down,
    t.next_is_paused
FROM ml_predictions_daily p
JOIN ml_tradeability_daily t
  ON p.trade_date = t.trade_date
 AND p.code = t.code
WHERE p.trade_date BETWEEN ? AND ?
  AND p.run_id = ?
  AND p.score_version = ?
```

如 `ml_tradeability_daily` 目前没有 `is_bse`，则通过 `detect_is_bse()` 识别。

### 达标标准

- 回测候选表包含所有交易和组合约束字段。
- ST、停牌、不可买、流动性不足、北交所均可被过滤。
- 不再只用裸预测表构建组合。
- 测试通过。

### 测试命令

```bash
python -m pytest \
  tests/test_backtest_candidate_join.py \
  tests/test_backtest_execution.py \
  -v
```

---

## Task 4.2: Use Config-Driven Portfolio v2 Constraints

### 当前问题

当前回测里存在：

```text
PortfolioConstraints(min_trade_score=-999.0)
```

这会绕过配置阈值，导致低分股票也可能进入组合。

### 修改文件

```text
scripts/run_ml_backtest.py
ml_stock_selector/portfolio/constraints.py
ml_stock_selector/portfolio/constructor.py
tests/test_portfolio_constructor.py
tests/test_backtest_candidate_join.py
```

### 配置建议

```toml
[portfolio.v2]
target_positions = 12
hard_max_positions = 15
max_industry_names = 3
max_unknown_industry_names = 1
max_new_entries_per_day = 4
min_trade_score = 0.80
min_core_pool_size = 1
min_candidate_pool_size = 5
allow_cash = true
min_adv20_amount = 50000000
exclude_bse = true
```

### 规则

1. `run_ml_backtest.py` 必须从配置加载 portfolio v2 参数。
2. 不允许硬编码 `min_trade_score=-999.0`。
3. 如果合格候选不足，可以空仓或轻仓，不强行买满。
4. UNKNOWN 行业最多 1 只，除非配置修改。
5. 北交所必须剔除。
6. 行业字段仅用于约束和报告，不用于模型特征。

### 达标标准

- 配置阈值实际生效。
- `min_trade_score` 调高后选股数量减少。
- `allow_cash=true` 时允许目标权重和小于 1。
- 北交所不进入 `ml_portfolio_targets_daily`。
- 测试通过。

### 测试命令

```bash
python -m pytest \
  tests/test_portfolio_constructor.py \
  tests/test_backtest_candidate_join.py \
  -v
```

---

## Task 4.3: Implement Core Pool + Candidate Pool Portfolio v2

### 目标

组合构建从“单一 trade_score 排序”升级为三模型 v2 推荐逻辑：

```text
core_pool + candidate_pool + trade_score_v2 排序 + 约束
```

### 修改文件

```text
ml_stock_selector/portfolio/constructor.py
ml_stock_selector/portfolio/constraints.py
tests/test_portfolio_constructor_v2.py
```

### 推荐规则

#### candidate_pool

```text
(
  absolute_rank_pct >= 0.70
  OR active_rank_pct >= 0.70
)
AND risk_rank_pct <= 0.65
AND trade_score_v2 >= min_trade_score
AND can_buy_next_open = true
AND is_st = false
AND is_paused = false
AND is_bse = false
```

#### core_pool

```text
absolute_rank_pct >= 0.80
AND active_rank_pct >= 0.75
AND risk_rank_pct <= 0.40
AND trade_score_v2 >= 0.80
AND can_buy_next_open = true
AND is_bse = false
```

### 输出字段

`ml_portfolio_targets_daily` 的 `entry_reason` 应包含：

```text
core_pool
candidate_pool
risk_filtered
low_trade_score
industry_limit
unknown_industry_limit
bse_excluded
```

### 达标标准

- core_pool 优先进入组合。
- core_pool 不足时从 candidate_pool 补充。
- candidate_pool 也不足时允许现金。
- 三模型交集不作为唯一策略，只作为 core_pool 强信号。
- 测试通过。

### 测试命令

```bash
python -m pytest tests/test_portfolio_constructor_v2.py -v
```

---

## Task 4.4: Produce Backtest Metrics per Fold and Run

### 当前问题

当前批量回测没有完整生成：

```text
ml_backtest_metrics
```

也没有 fold/run 级指标。

### 修改文件

```text
ml_stock_selector/backtest/metrics.py
ml_stock_selector/backtest/reports.py
ml_stock_selector/backtest/engine.py
scripts/run_ml_backtest.py
tests/test_backtest_metrics.py
tests/test_walkforward.py
```

### 指标字段

`ml_backtest_metrics` 至少包含：

```text
run_id
fold_id
strategy_id
score_version
start_date
end_date

annual_return
total_return
max_drawdown
calmar_like
turnover
win_rate
avg_holding_days
cash_ratio_avg
empty_day_count
empty_day_ratio
unknown_industry_weight_avg
unknown_industry_weight_max
core_pool_size_avg
candidate_pool_size_avg
risk_rejected_count
bse_excluded_count
created_at
```

### 达标标准

- 每个 fold 都有一条或多条 metrics。
- 总 run 有汇总 metrics。
- 空仓比例可见。
- 北交所剔除数量可见。
- UNKNOWN 行业暴露可见。
- 测试通过。

### 测试命令

```bash
python -m pytest \
  tests/test_backtest_metrics.py \
  tests/test_walkforward.py \
  -v
```

---

# Phase 5: Benchmark Audit Layer

## Phase 5 目标

补齐 benchmark 表未接入生产链的问题。此任务不是训练 P0 阻塞，但属于三模型 v2 可审计性的必要增强。

---

## Task 5.1: Materialize Market and Industry Benchmark Tables

### 当前问题

SQL 中已有：

```text
ml_market_benchmark_daily
ml_industry_benchmark_daily
```

但当前标签脚本没有写这两张表。`label_builder.py` 是内部计算 benchmark，虽然能用，但不可审计。

### 修改文件

```text
ml_stock_selector/benchmarks.py
scripts/build_ml_benchmarks.py
ml_stock_selector/label_builder.py
tests/test_ml_benchmarks.py
tests/test_ml_labels_excess_return.py
```

### 表定义建议

#### ml_market_benchmark_daily

```text
trade_date
benchmark_id
open
high
low
close
prev_close
ret_1d
ret_5d
ret_10d
source_row_count
generated_at
```

#### ml_industry_benchmark_daily

```text
trade_date
industry_code
industry_name
open
high
low
close
prev_close
ret_1d
ret_5d
ret_10d
source_row_count
generated_at
```

### 数据来源

优先读取：

```text
vpa_scope_bars_daily
```

如果不存在，可 fallback 到当前内部计算逻辑。

### label_builder 规则

1. benchmark 表存在时优先读取。
2. benchmark 表不存在时 fallback 到内部计算。
3. active label 计算结果应与内部计算结果在 fixture 中一致。
4. UNKNOWN 行业的 `industry_ret` 可以为空，active score fallback 到 market excess。

### 达标标准

- benchmark 表可生成。
- label_builder 能读 benchmark 表。
- active label 可追溯 market/industry benchmark。
- fallback 模式仍然可用。
- 测试通过。

### 测试命令

```bash
python -m pytest \
  tests/test_ml_benchmarks.py \
  tests/test_ml_labels_excess_return.py \
  -v
```

---

# Phase 6: Daily Signal v2

## Phase 6 目标

修复日信号，使其真正使用 active 三模型、VPA-ML 宇宙过滤和 portfolio v2 约束。

---

## Task 6.1: Load Active Production Three Models for Daily Signal

### 修改文件

```text
ml_stock_selector/serving/daily_signal.py
ml_stock_selector/serving/artifact_loader.py
tests/test_daily_signal.py
```

### 规则

1. daily signal 必须加载 production active：
   ```text
   absolute_ranker
   active_ranker
   risk_model
   ```
2. 如果任一模型缺失，daily signal 应失败并给出明确错误。
3. daily signal 不使用 walk-forward fold 模型。
4. daily signal 不依赖 `ml_labels_daily`。

### 达标标准

- 三个 active model 缺一时报错。
- 三个模型齐全时可生成预测。
- 日信号不读取未来标签。
- 测试通过。

### 测试命令

```bash
python -m pytest tests/test_daily_signal.py -v
```

---

## Task 6.2: Apply VPA-ML Universe and Portfolio v2 in Daily Signal

### 修改文件

```text
ml_stock_selector/serving/daily_signal.py
ml_stock_selector/portfolio/constructor.py
tests/test_daily_signal.py
tests/test_prediction_exclude_bse.py
```

### 规则

daily signal 流程：

```text
1. 读取 as_of_date 的 ml_feature_mart_daily
2. join ml_tradeability_daily
3. exclude_bse = true
4. 加载三模型
5. 预测 absolute / active / risk
6. 计算 trade_score_v2
7. 应用 portfolio v2 约束
8. 输出 target portfolio
```

输出报告至少包含：

```text
as_of_date
universe_name
excluded_bse_count
input_candidates
post_filter_candidates
core_pool_size
candidate_pool_size
selected_count
cash_weight
blocked_reasons
```

### 达标标准

- 北交所不进入日信号候选或目标组合。
- 分数不够时允许空仓。
- daily signal 目标组合数量不超过 hard_max_positions。
- 输出可解释 blocked reason。
- 测试通过。

### 测试命令

```bash
python -m pytest \
  tests/test_daily_signal.py \
  tests/test_prediction_exclude_bse.py \
  -v
```

---

# Phase 7: End-to-End Acceptance

## Phase 7 目标

完整验收三模型 v2 闭环。

---

## Task 7.1: Add End-to-End Smoke Test

### 修改文件

```text
tests/test_three_model_v2_end_to_end.py
```

### 测试流程

fixture 中构造：

```text
100 trading days
20 stocks
其中 2 只北交所
包含 ST / paused / missing_limit / UNKNOWN industry
包含 abs_label / active_label / risk_label
```

测试：

```text
1. build samples
2. exclude BSE
3. train three models
4. fold predict
5. compute trade_score_v2
6. build portfolio targets
7. run simplified backtest
8. write metrics
```

### 达标标准

- 端到端流程跑通。
- 北交所样本不进入训练、不进入预测、不进入组合。
- 三模型都产生预测分数。
- portfolio 不超过 hard_max_positions。
- metrics 落库。
- 测试通过。

### 测试命令

```bash
python -m pytest tests/test_three_model_v2_end_to_end.py -v
```

---

## Task 7.2: Final Verification Commands

### 单元测试

```bash
python -m pytest \
  tests/test_ml_config.py \
  tests/test_universe_filter.py \
  tests/test_feature_matrix.py \
  tests/test_no_industry_training_features.py \
  tests/test_sample_builder.py \
  tests/test_training_samples_exclude_bse.py \
  tests/test_active_ranker.py \
  tests/test_risk_model_probability.py \
  tests/test_ml_prediction.py \
  tests/test_three_model_prediction_merge.py \
  tests/test_prediction_exclude_bse.py \
  tests/test_ml_scoring.py \
  tests/test_backtest_candidate_join.py \
  tests/test_portfolio_constructor.py \
  tests/test_portfolio_constructor_v2.py \
  tests/test_backtest_metrics.py \
  tests/test_daily_signal.py \
  -v
```

### 集成测试

```bash
python -m pytest \
  tests/test_walkforward.py \
  tests/test_three_model_v2_end_to_end.py \
  -v
```

### 手动验收 SQL

#### 1. 2015 是否进入训练范围

```sql
select
  min(trade_date) as min_date,
  max(trade_date) as max_date,
  count(*) as rows
from ml_feature_mart_daily
where trade_date >= '2015-01-05';
```

达标：

```text
min_date = 2015-01-05
rows > 0
```

#### 2. 北交所是否进入预测

```sql
select count(*) as bse_predictions
from ml_predictions_daily p
join ml_tradeability_daily t
  on p.trade_date = t.trade_date
 and p.code = t.code
where p.score_version = 'v2_three_model'
  and (
    coalesce(t.is_bse, false) = true
    or p.code like '920%'
  );
```

达标：

```text
bse_predictions = 0
```

如果采用保留预测但标记不可选模式，则达标条件改为：

```text
eligible_universe = false
且不能进入 ml_portfolio_targets_daily
```

#### 3. 组合是否包含北交所

```sql
select count(*) as bse_targets
from ml_portfolio_targets_daily pt
join ml_tradeability_daily t
  on pt.trade_date = t.trade_date
 and pt.code = t.code
where (
    coalesce(t.is_bse, false) = true
    or pt.code like '920%'
);
```

达标：

```text
bse_targets = 0
```

#### 4. 预测是否有三模型分数

```sql
select
  count(*) as rows,
  sum(case when absolute_rank_pct is null then 1 else 0 end) as null_abs,
  sum(case when active_rank_pct is null then 1 else 0 end) as null_active,
  sum(case when risk_rank_pct is null then 1 else 0 end) as null_risk,
  sum(case when trade_score_v2 is null then 1 else 0 end) as null_trade_score
from ml_predictions_daily
where score_version = 'v2_three_model';
```

达标：

```text
null_abs = 0
null_active = 0
null_risk = 0
null_trade_score = 0
```

#### 5. 回测指标是否落库

```sql
select
  run_id,
  fold_id,
  count(*) as metric_rows
from ml_backtest_metrics
where score_version = 'v2_three_model'
group by run_id, fold_id
order by run_id, fold_id;
```

达标：

```text
每个 fold 至少 1 条 metrics
```

---

# Phase 8: Recommended Execution Order

建议 Codex 或人工按以下顺序实施：

```text
1. Task 1.1: universe 配置
2. Task 1.2: BSE 识别与过滤
3. Task 1.3: walk-forward 多 fold 配置，2015 纳入训练

4. Task 2.2: training samples 支持 universe filter
5. Task 2.3: feature matrix 排除行业/交易/宇宙 metadata
6. Task 2.1: three-model walk-forward runner

7. Task 3.1: batch predict 支持 fold/run/date/model scope
8. Task 3.2: batch predict 剔除 BSE
9. Task 3.3: 落地 trade_score_v2

10. Task 4.1: backtest candidate join tradeability metadata
11. Task 4.2: portfolio v2 配置化约束
12. Task 4.3: core/candidate pool
13. Task 4.4: backtest metrics 落库

14. Task 5.1: benchmark audit tables

15. Task 6.1: daily signal 加载 active 三模型
16. Task 6.2: daily signal 使用 universe + portfolio v2

17. Task 7.1: end-to-end smoke
18. Task 7.2: final verification
```

---

# Definition of Done

本任务文件完成的最终标准：

```text
1. 2015-01-05 起的样本能进入训练、walk-forward 和回测。
2. 北交所股票不进入训练样本。
3. 北交所股票不进入 v2 预测、组合、回测、日信号。
4. walk-forward 每个 fold 都训练 absolute_ranker / active_ranker / risk_model。
5. fold 模型注册但不激活为 production active。
6. batch predict 支持 run_id / fold_id / date range / 三模型 model_id。
7. ml_predictions_daily 或组件表中可追溯三模型分数和最终 trade_score_v2。
8. 回测候选表合并 tradeability metadata。
9. ST、停牌、可买、流动性、行业、UNKNOWN、BSE 约束实际生效。
10. portfolio v2 使用配置阈值，不再硬编码 min_trade_score=-999。
11. 回测产出 fold/run 级 metrics。
12. daily signal 只使用 production active 三模型。
13. 行业信息不进入 feature matrix。
14. 所有新增测试通过。
```

---

# Commit Discipline

每个 task 单独 commit。

Commit message 格式：

```text
<imperative summary>

<why this task exists>

Constraint: Three-model v2 learns price-volume/VPA patterns, excludes BSE in VPA-ML, and keeps industry as label/portfolio metadata only
Confidence: <high|medium|low>
Scope-risk: <narrow|moderate|broad>
Tested: <exact pytest command>
```

示例：

```bash
git commit -m "Add VPA-ML universe filter excluding BSE" -m "Training, prediction, portfolio construction, backtest, and daily signal must operate on the same CN A-share ex-BSE universe so BSE liquidity and trading-regime differences do not contaminate the three-model strategy.

Constraint: Three-model v2 learns price-volume/VPA patterns, excludes BSE in VPA-ML, and keeps industry as label/portfolio metadata only
Confidence: high
Scope-risk: moderate
Tested: python -m pytest tests/test_universe_filter.py tests/test_training_samples_exclude_bse.py -v"
```
