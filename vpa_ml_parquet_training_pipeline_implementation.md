# VPA-ML Parquet Feature Store Training Pipeline Implementation Plan

> **适用项目**：`miraclecn/volume-price-analysis`  
> **适用模块**：`ml_stock_selector`  
> **目标**：将当前基于 `features_json + pandas 全量展开` 的训练管线，升级为 **Parquet Feature Store + Fold Matrix Cache + 分块预测 + DuckDB SQL 排名**，使三模型 walk-forward 能在 32G 内存机器上稳定运行。
>
> 本文档用于 Codex `/goal`、人工实施、代码评审与验收。

---

## 0. 背景与问题

当前三张核心数据表已经生成：

```text
ml_tradeability_daily：11,216,101 行，2015-01-05 ~ 2026-05-29
ml_feature_mart_daily：11,216,101 行，2015-01-05 ~ 2026-05-29
ml_labels_daily：49,354,556 行，2015-01-05 ~ 2026-05-28
```

当前 walk-forward 训练在 `wf_2020` 单 fold 上出现 OOM：

```text
退出码：137
原因：系统 kill，基本可判定为 OOM
机器内存：32G
```

根因不是 LightGBM 参数，而是训练管线的数据访问方式：

```text
select * 全量读取
        ↓
pandas 全量载入
        ↓
features_json 全量展开
        ↓
内存爆炸
```

`wf_2020` 仅训练段就约 277 万行，`features_json` 平均约 5KB，原始 JSON 体量接近 18GB；进入 pandas object、dict、DataFrame、矩阵之后会远超 32G。

---

## 1. 改造目标

### 1.1 总体链路

```text
ml_feature_mart_daily.features_json
        ↓ 一次性分块解析
Parquet Feature Store
        ↓ fold/date predicate pushdown
Fold Matrix Cache
        ↓ 三模型共享 X
Absolute Ranker / Active Ranker / Risk Model
        ↓ 分块预测
ml_prediction_raw_daily
        ↓ DuckDB SQL rank_pct / trade_score_v2
ml_predictions_daily
        ↓ JOIN ml_tradeability_daily
portfolio / backtest / daily signal
```

### 1.2 三模型定义保持不变

```text
Absolute Ranker:
  学习价格/成交量/VPA 量价关系 → 未来绝对收益排名

Active Ranker:
  学习价格/成交量/VPA 量价关系 → 未来相对大盘/行业的主动收益排名

Risk Model:
  学习价格/成交量/VPA 量价关系 → 未来回撤风险概率
```

### 1.3 训练特征边界

行业、交易所、北交所、ST、停牌、可买卖字段是 **metadata / 过滤 / 组合约束字段**，不得进入训练特征。

禁止进入模型矩阵：

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

### 1.4 VPA-ML 策略宇宙

```text
2014 = warm-up
2015-01-05 起 = 正式训练 / walk-forward / 回测样本
北交所 = 在 VPA-ML 阶段剔除
```

---

# 2. P0：让 wf_2020 在 32G 机器上跑通

P0 是最小可运行改造。目标是先让单 fold `wf_2020` 不 OOM，并完整训练三模型、预测、落库。

---

## Task P0.1: 新增 Parquet Feature Store Exporter

### 目标

把 `ml_feature_mart_daily.features_json` 分块解析为列式 Parquet，避免 walk-forward 每次从 JSON 大字段开始解析。

### 新增文件

```text
ml_stock_selector/feature_store.py
scripts/export_ml_feature_store.py
tests/test_feature_store_export.py
tests/test_feature_store_schema.py
```

### 输入

```text
ml_feature_mart_daily
```

必要字段：

```text
trade_date
code
feature_set_id
features_json
```

### 输出目录

```text
outputs/ml/feature_store/
  dataset_version=v2_pv_only_001/
    feature_set_id=vpa_d_sequence/
      year=2015/
        month=01/
          part-000.parquet
          part-001.parquet
      year=2015/
        month=02/
          part-000.parquet
      ...
    feature_schema.json
    _metadata.json
```

### Parquet 字段要求

每个 Parquet part 至少包含：

```text
trade_date
code
feature_set_id
feature_schema_version
<feature columns...>
```

其中 `<feature columns...>` 只包含模型训练特征。

### 明确排除字段

无论这些字段是否出现在 `features_json` 中，都不得写入训练 Parquet feature columns：

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

### dtype 规则

```text
数值特征：float32
布尔特征：uint8 或 bool
类别特征：P0 阶段默认排除，P1/P2 再考虑稳定编码
trade_date：date 或 string，需可被 DuckDB / pyarrow filter
code：string
```

### P0 类别特征处理建议

为了尽快跑通 wf_2020：

```text
P0 只导出 numeric features；
raw_label / sequence_pattern 等字符串类别特征暂时排除；
后续 P1/P2 通过 feature allowlist + category mapping 稳定纳入。
```

### CLI

```bash
python scripts/export_ml_feature_store.py \
  --ml-db outputs/ml/ml.duckdb \
  --output-dir outputs/ml/feature_store \
  --dataset-version v2_pv_only_001 \
  --feature-set-id vpa_d_sequence \
  --start-date 2015-01-05 \
  --end-date 2026-05-29 \
  --chunk-size 20000 \
  --row-group-size 50000 \
  --compression zstd
```

### 实现要点

1. SQL 侧按日期和 `feature_set_id` 过滤。
2. 每次只读取 `chunk_size` 行。
3. chunk 内解析 `features_json`。
4. 根据 denylist 删除 metadata 字段。
5. 尽量使用 `float32`。
6. 按 `year/month` 分区写 Parquet。
7. 每个 chunk 写完释放内存。
8. 导出完成后写 `feature_schema.json` 和 `_metadata.json`。

### `_metadata.json` 示例

```json
{
  "dataset_version": "v2_pv_only_001",
  "feature_set_id": "vpa_d_sequence",
  "source_table": "ml_feature_mart_daily",
  "source_start_date": "2015-01-05",
  "source_end_date": "2026-05-29",
  "row_count": 11216101,
  "dtype_policy": "float32",
  "compression": "zstd",
  "excluded_metadata_columns": [
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
    "can_sell_next_open"
  ],
  "created_at": "<timestamp>"
}
```

### `feature_schema.json` 示例

```json
{
  "feature_set_id": "vpa_d_sequence",
  "dataset_version": "v2_pv_only_001",
  "schema_version": "v2_pv_only_001",
  "numeric_columns": [
    "ret_1d",
    "range_pct",
    "body_pct"
  ],
  "categorical_columns": [],
  "fill_values": {
    "numeric": 0.0
  },
  "excluded_metadata_columns": [
    "industry_code",
    "industry_name",
    "is_bse",
    "is_st",
    "is_paused",
    "can_buy_next_open"
  ]
}
```

### 验收标准

- 可以从 2015-01-05 导出到 2026-05-29。
- 导出过程不 OOM。
- Parquet 文件按 `feature_set_id/year/month` 分区。
- `_metadata.json.row_count` 与源表过滤后行数一致或在允许误差内。
- `feature_schema.json` 存在且列名稳定。
- Parquet 中不包含行业、交易所、北交所、ST、停牌、可交易性字段。
- 至少可用 DuckDB 成功读取某一年数据。

### 测试命令

```bash
python -m pytest \
  tests/test_feature_store_export.py \
  tests/test_feature_store_schema.py \
  -v
```

### 手动验收 SQL

```sql
select count(*) as rows
from read_parquet('outputs/ml/feature_store/dataset_version=v2_pv_only_001/feature_set_id=vpa_d_sequence/year=2015/month=*/*.parquet');
```

---

## Task P0.2: 新增 Feature Store Reader

### 目标

提供统一读取 Parquet Feature Store 的接口，支持 fold 日期过滤、列过滤和分块读取。

### 新增文件

```text
ml_stock_selector/feature_store_reader.py
tests/test_feature_store_reader.py
```

### 接口

```python
@dataclass(frozen=True)
class FeatureStoreSpec:
    feature_store_dir: str
    dataset_version: str
    feature_set_id: str
    schema_version: str | None = None

def iter_feature_store_batches(
    spec: FeatureStoreSpec,
    start_date: str,
    end_date: str,
    columns: list[str] | None = None,
    batch_size: int = 50000,
) -> Iterator[pd.DataFrame]:
    ...

def load_feature_schema(spec: FeatureStoreSpec) -> FeatureSchema:
    ...
```

### 规则

1. 必须按 `start_date/end_date` 过滤。
2. 必须支持只读取部分列。
3. 默认读取 `feature_schema.numeric_columns + ["trade_date", "code"]`。
4. 不得读取无关 metadata。
5. 读取时保持列顺序与 `feature_schema` 一致。

### 验收标准

- 能读取 `wf_2020` train/valid/test 所需日期。
- 能按 batch 返回 DataFrame。
- 返回列顺序稳定。
- 读取时不包含 denylist 字段。
- 测试通过。

### 测试命令

```bash
python -m pytest tests/test_feature_store_reader.py -v
```

---

## Task P0.3: `run_ml_walkforward.py` 支持 fold-id 和 feature-store 输入

### 目标

修复 walk-forward 默认全量读取的问题，支持单 fold 运行和 Parquet Feature Store。

### 修改文件

```text
scripts/run_ml_walkforward.py
ml_stock_selector/backtest/walkforward.py
tests/test_walkforward_cli.py
tests/test_walkforward.py
```

### 新增 CLI 参数

```bash
--fold-id wf_2020
--feature-store-dir outputs/ml/feature_store
--feature-store-version v2_pv_only_001
--use-feature-store true
--matrix-cache-dir outputs/ml/cache/folds
```

### 示例命令

```bash
python scripts/run_ml_walkforward.py \
  --config config/ml_walkforward.toml \
  --ml-db outputs/ml/ml.duckdb \
  --run-id wf_three_model_v2_parquet_001 \
  --fold-id wf_2020 \
  --feature-store-dir outputs/ml/feature_store \
  --feature-store-version v2_pv_only_001 \
  --use-feature-store true \
  --matrix-cache-dir outputs/ml/cache/folds \
  --feature-set-id vpa_d_sequence \
  --horizon-d 5 \
  --label-base from_next_open \
  --score-version v2_three_model
```

### 规则

1. 如果传入 `--fold-id`，只运行该 fold。
2. 如果未传入 `--fold-id`，按配置运行全部 fold。
3. `--use-feature-store true` 时，不得读取 `ml_feature_mart_daily.features_json`。
4. runner 内所有数据读取必须带 fold 日期范围。
5. 禁止 `select * from ml_feature_mart_daily`。
6. 运行日志必须打印：
   ```text
   fold_id
   train/valid/test date range
   feature_store_version
   estimated train/valid/test rows
   exclude_bse
   ```

### 验收标准

- `--fold-id wf_2020` 只运行 wf_2020。
- 不再全量读取 11M feature_mart。
- 不再全量读取 49M labels。
- 不再直接展开 features_json。
- 测试通过。

### 测试命令

```bash
python -m pytest \
  tests/test_walkforward_cli.py \
  tests/test_walkforward.py \
  -v
```

---

## Task P0.4: 新增 Fold Matrix Cache

### 目标

将每个 fold 的训练/验证/测试矩阵落地到磁盘，三模型共享同一套 `X`，避免重复解析和重复展开。

### 新增文件

```text
ml_stock_selector/matrix_cache.py
tests/test_matrix_cache.py
```

### 输出目录

```text
outputs/ml/cache/folds/
  run_id=wf_three_model_v2_parquet_001/
    fold_id=wf_2020/
      X_train.npz
      X_valid.npz
      X_test.npz

      y_abs_train.npy
      y_abs_valid.npy
      y_active_train.npy
      y_active_valid.npy
      y_risk_train.npy
      y_risk_valid.npy

      group_train.npy
      group_valid.npy

      metadata_train.parquet
      metadata_valid.parquet
      metadata_test.parquet

      feature_schema.json
      manifest.json
```

### 新增接口

```python
@dataclass(frozen=True)
class FoldMatrixCache:
    run_id: str
    fold_id: str
    cache_dir: Path
    x_train_path: Path
    x_valid_path: Path
    x_test_path: Path
    y_abs_train_path: Path
    y_abs_valid_path: Path
    y_active_train_path: Path
    y_active_valid_path: Path
    y_risk_train_path: Path
    y_risk_valid_path: Path
    group_train_path: Path
    group_valid_path: Path
    metadata_train_path: Path
    metadata_valid_path: Path
    metadata_test_path: Path
    feature_schema_path: Path
    manifest_path: Path

def build_fold_matrix_cache(
    con: duckdb.DuckDBPyConnection,
    feature_store_spec: FeatureStoreSpec,
    fold_config: WalkForwardFoldConfig,
    run_id: str,
    feature_set_id: str,
    horizon_d: int,
    label_base: str,
    universe_config: UniverseConfig,
    cache_root: str,
    batch_size: int = 50000,
) -> FoldMatrixCache:
    ...
```

### 数据来源

```text
Features:  Parquet Feature Store
Labels:    ml_labels_daily
Metadata:  ml_tradeability_daily
```

### label 字段

```text
y_abs    = rank_label_abs
y_active = rank_label_active
y_risk   = risk_label
```

### 过滤规则

```text
trade_date >= 2015-01-05
fold train/valid/test 日期范围
feature_set_id = vpa_d_sequence
horizon_d = 5
label_base = from_next_open
exclude_bse = true
rank_label_abs not null
rank_label_active not null
risk_label not null
```

### group 规则

LightGBM Ranker 的 group 由 `trade_date` 构造：

```text
group_train = 每个 trade_date 的样本数
group_valid = 每个 trade_date 的样本数
```

metadata 中保留：

```text
trade_date
code
industry_code
industry_name
is_bse
is_st
is_paused
adv20_amount
can_buy_next_open
```

但这些 metadata 不进入 `X`。

### 格式要求

```text
X_train / X_valid / X_test:
  scipy.sparse.csr_matrix
  dtype=float32
  保存为 .npz

y:
  numpy array
  保存为 .npy

group:
  numpy array
  保存为 .npy

metadata:
  parquet
```

### `manifest.json`

```json
{
  "run_id": "wf_three_model_v2_parquet_001",
  "fold_id": "wf_2020",
  "status": "matrix_built",
  "feature_store_version": "v2_pv_only_001",
  "feature_set_id": "vpa_d_sequence",
  "horizon_d": 5,
  "label_base": "from_next_open",
  "train_rows": 2771471,
  "valid_rows": 0,
  "test_rows": 945983,
  "exclude_bse": true,
  "created_at": "<timestamp>"
}
```

### 验收标准

- `wf_2020` 能生成 fold cache。
- cache 生成过程不 OOM。
- 三模型共享同一套 X。
- 北交所样本不在 metadata_train/test 中。
- `group_train.sum() == X_train.shape[0]`。
- `y_abs_train.shape[0] == X_train.shape[0]`。
- 测试通过。

### 测试命令

```bash
python -m pytest tests/test_matrix_cache.py -v
```

---

## Task P0.5: 三模型训练改为读取 Fold Matrix Cache

### 目标

训练阶段不再读取数据库特征表，也不再解析 JSON。三个模型都从 fold cache 读取。

### 修改文件

```text
ml_stock_selector/models/alpha_ranker.py
ml_stock_selector/models/active_ranker.py
ml_stock_selector/models/risk_model.py
ml_stock_selector/backtest/walkforward.py
tests/test_train_from_matrix_cache.py
tests/test_walkforward.py
```

### 新增接口

```python
def train_three_models_from_fold_cache(
    cache: FoldMatrixCache,
    config: MLConfig,
    artifact_dir: str,
) -> ThreeModelFoldArtifacts:
    ...
```

### 训练规则

Absolute Ranker：

```text
X_train + y_abs_train + group_train
X_valid + y_abs_valid + group_valid
```

Active Ranker：

```text
X_train + y_active_train + group_train
X_valid + y_active_valid + group_valid
```

Risk Model：

```text
X_train + y_risk_train
X_valid + y_risk_valid
```

### LightGBM 推荐参数

```toml
[model.lightgbm_runtime]
num_threads = 8
force_col_wise = true
max_bin = 63
num_leaves = 31
min_data_in_leaf = 500
feature_fraction = 0.8
bagging_fraction = 0.8
bagging_freq = 1
```

### registry 规则

fold 模型写入 `ml_model_registry`，但必须：

```text
is_active = false
```

记录字段：

```text
run_id
fold_id
model_type
feature_store_version
feature_schema_uri
artifact_uri
train_start
train_end
valid_start
valid_end
test_start
test_end
```

### 验收标准

- 三模型训练不读取 `ml_feature_mart_daily`。
- 三模型训练不解析 `features_json`。
- 每个 fold 生成 3 个 model_id。
- fold 模型不激活为 production active。
- artifact 和 feature_schema 可复用。
- 测试通过。

### 测试命令

```bash
python -m pytest \
  tests/test_train_from_matrix_cache.py \
  tests/test_walkforward.py \
  -v
```

---

## Task P0.6: 分块预测 + DuckDB SQL rank_pct/trade_score_v2

### 目标

避免 test 全量进 pandas 排名。预测分块写 raw 表，然后用 DuckDB SQL 计算横截面分位数和 `trade_score_v2`。

### 修改文件

```text
ml_stock_selector/prediction.py
ml_stock_selector/scoring.py
scripts/run_ml_batch_predict.py
tests/test_chunked_prediction.py
tests/test_sql_prediction_ranking.py
```

### 新增 raw 表

```text
ml_prediction_raw_daily
```

字段：

```text
trade_date
code
run_id
fold_id
score_version
feature_set_id
horizon_d

absolute_model_id
active_model_id
risk_model_id

absolute_score
active_score
risk_prob

generated_at
```

### 最终表

`ml_predictions_daily` 至少包含：

```text
trade_date
code
run_id
fold_id
score_version
feature_set_id
horizon_d

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

### 预测流程

```text
1. 读取 X_test.npz 和 metadata_test.parquet
2. 按 chunk_size 切分 X_test
3. 三模型预测 raw scores
4. 分块写入 ml_prediction_raw_daily
5. DuckDB SQL 计算 absolute_rank_pct / active_rank_pct / risk_rank_pct
6. DuckDB SQL 计算 trade_score_v2
7. 写入 ml_predictions_daily
```

### rank_pct SQL

```sql
percent_rank() over (
  partition by trade_date
  order by absolute_score
) as absolute_rank_pct
```

```sql
percent_rank() over (
  partition by trade_date
  order by active_score
) as active_rank_pct
```

```sql
percent_rank() over (
  partition by trade_date
  order by risk_prob
) as risk_rank_pct
```

### trade_score_v2

默认公式：

```text
trade_score_v2 =
0.55 * absolute_rank_pct
+ 0.35 * active_rank_pct
- 0.25 * risk_rank_pct
```

### 验收标准

- 预测阶段不一次性展开 test features。
- `ml_prediction_raw_daily` 有 raw scores。
- `ml_predictions_daily` 有三模型 rank_pct 和 trade_score_v2。
- 同一 `trade_date` 内 rank_pct 分布在 0~1。
- 北交所不进入预测。
- 测试通过。

### 测试命令

```bash
python -m pytest \
  tests/test_chunked_prediction.py \
  tests/test_sql_prediction_ranking.py \
  -v
```

---

## Task P0.7: wf_2020 单 fold 验收

### 命令顺序

#### 1. 导出 feature store

```bash
python scripts/export_ml_feature_store.py \
  --ml-db outputs/ml/ml.duckdb \
  --output-dir outputs/ml/feature_store \
  --dataset-version v2_pv_only_001 \
  --feature-set-id vpa_d_sequence \
  --start-date 2015-01-05 \
  --end-date 2026-05-29 \
  --chunk-size 20000 \
  --row-group-size 50000 \
  --compression zstd
```

#### 2. 跑 wf_2020

```bash
python scripts/run_ml_walkforward.py \
  --config config/ml_walkforward.toml \
  --ml-db outputs/ml/ml.duckdb \
  --run-id wf_three_model_v2_parquet_001 \
  --fold-id wf_2020 \
  --feature-store-dir outputs/ml/feature_store \
  --feature-store-version v2_pv_only_001 \
  --use-feature-store true \
  --matrix-cache-dir outputs/ml/cache/folds \
  --feature-set-id vpa_d_sequence \
  --horizon-d 5 \
  --label-base from_next_open \
  --score-version v2_three_model
```

### 达标标准

- 进程不 OOM。
- 生成 fold cache。
- 训练出三模型。
- 写入 fold model registry。
- 写入 raw predictions。
- 写入 `ml_predictions_daily`。
- `ml_predictions_daily` 中无北交所预测。
- `trade_score_v2` 非空。
- 如果回测链路已接通，则生成 fold metrics。

### SQL 验收

```sql
select
  run_id,
  fold_id,
  count(*) as rows,
  min(trade_date) as min_date,
  max(trade_date) as max_date
from ml_predictions_daily
where run_id = 'wf_three_model_v2_parquet_001'
group by run_id, fold_id;
```

```sql
select
  count(*) as rows,
  sum(case when absolute_rank_pct is null then 1 else 0 end) as null_abs,
  sum(case when active_rank_pct is null then 1 else 0 end) as null_active,
  sum(case when risk_rank_pct is null then 1 else 0 end) as null_risk,
  sum(case when trade_score_v2 is null then 1 else 0 end) as null_trade_score
from ml_predictions_daily
where run_id = 'wf_three_model_v2_parquet_001';
```

期望：

```text
null_abs = 0
null_active = 0
null_risk = 0
null_trade_score = 0
```

---

# 3. P1：完整 walk-forward 可重复、可恢复、可审计

P1 在 P0 跑通 wf_2020 后进行。

---

## Task P1.1: Feature Store 版本化与一致性校验

### 目标

让 Feature Store 成为可审计数据集。

### 修改文件

```text
ml_stock_selector/feature_store.py
ml_stock_selector/feature_store_reader.py
tests/test_feature_store_metadata.py
```

### 要求

1. `_metadata.json` 记录：
   ```text
   dataset_version
   feature_set_id
   row_count
   min_date
   max_date
   created_at
   source_table
   source_db
   schema_hash
   ```
2. `feature_schema.json` 有 `schema_hash`。
3. 读取时校验 schema_hash。
4. 如果 schema 不匹配，直接报错。

### 达标标准

- Feature Store 可以被唯一版本定位。
- 模型 registry 记录 `feature_store_version`。
- fold manifest 记录 `feature_store_version`。
- 测试通过。

---

## Task P1.2: Fold Manifest and Resume

### 目标

walk-forward 支持中断恢复。

### 修改文件

```text
ml_stock_selector/matrix_cache.py
ml_stock_selector/backtest/walkforward.py
tests/test_walkforward_resume.py
```

### manifest 状态机

```text
pending
matrix_built
models_trained
predicted
backtested
metrics_written
failed
```

### 规则

1. 如果 `manifest.status = matrix_built`，可跳过 matrix 构建。
2. 如果 `models_trained`，可跳过训练进入预测。
3. 如果 `predicted`，可跳过预测进入回测。
4. 失败时写入：
   ```text
   status = failed
   failed_at
   error_message
   ```

### 达标标准

- 人为中断后可继续。
- 不重复解析 Parquet。
- 不重复训练已完成模型，除非传入 `--force`。
- 测试通过。

---

## Task P1.3: Full Multi-Fold Walk-Forward

### 目标

跑通所有正式 fold：

```text
wf_2020
wf_2021
wf_2022
wf_2023
wf_2024
wf_2025
wf_2026_ytd
```

### 命令

```bash
python scripts/run_ml_walkforward.py \
  --config config/ml_walkforward.toml \
  --ml-db outputs/ml/ml.duckdb \
  --run-id wf_three_model_v2_parquet_full_001 \
  --feature-store-dir outputs/ml/feature_store \
  --feature-store-version v2_pv_only_001 \
  --use-feature-store true \
  --matrix-cache-dir outputs/ml/cache/folds \
  --feature-set-id vpa_d_sequence \
  --horizon-d 5 \
  --label-base from_next_open \
  --score-version v2_three_model
```

### 达标标准

- 所有 fold 都有 manifest。
- 所有 fold 都完成 `metrics_written`，或至少到 `predicted`。
- 每个 fold 都有三模型 model_id。
- 每个 fold 都有预测行。
- 不 OOM。
- 可恢复。

### SQL 验收

```sql
select
  run_id,
  fold_id,
  count(*) as prediction_rows,
  min(trade_date) as min_date,
  max(trade_date) as max_date
from ml_predictions_daily
where run_id = 'wf_three_model_v2_parquet_full_001'
group by run_id, fold_id
order by fold_id;
```

---

## Task P1.4: Backtest Candidate Join and Metrics

### 目标

完成三模型回测闭环，确保回测候选 join `ml_tradeability_daily`。

### 修改文件

```text
ml_stock_selector/backtest/data_access.py
ml_stock_selector/backtest/engine.py
ml_stock_selector/backtest/metrics.py
scripts/run_ml_backtest.py
tests/test_backtest_candidate_join.py
tests/test_backtest_metrics.py
```

### 规则

回测候选必须包含：

```text
industry_code
industry_name
is_st
is_paused
is_bse
adv20_amount
can_buy_next_open
can_sell_next_open
next_open
next_limit_up
next_limit_down
next_is_paused
```

### 过滤规则

```text
exclude_bse = true
is_st = false
is_paused = false
can_buy_next_open = true
adv20_amount >= min_adv20_amount
trade_score_v2 >= min_trade_score
```

### metrics

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
empty_day_ratio
cash_ratio_avg
bse_excluded_count
unknown_industry_weight_avg
core_pool_size_avg
candidate_pool_size_avg
created_at
```

### 达标标准

- 回测不用裸 predictions。
- 回测约束实际生效。
- 北交所不进入 portfolio targets。
- 每个 fold 有 metrics。
- 测试通过。

---

# 4. P2：Parquet Feature Store 成为正式生产路径

P2 在 P1 稳定后进行。

---

## Task P2.1: Feature Mart 生成时直接写 Parquet

### 目标

不再依赖后处理 export，从 feature mart 构建阶段直接写 Parquet Feature Store。

### 修改文件

```text
ml_stock_selector/feature_mart.py
scripts/run_ml_feature_mart.py
ml_stock_selector/feature_store.py
tests/test_feature_mart_writes_feature_store.py
```

### CLI

```bash
python scripts/run_ml_feature_mart.py \
  --config config/ml_default.toml \
  --write-duckdb true \
  --write-feature-store true \
  --feature-store-dir outputs/ml/feature_store \
  --dataset-version v2_pv_only_001
```

### 达标标准

- feature_mart 生成时同步写 Parquet。
- `features_json` 可继续写入 DuckDB 作为 legacy / audit。
- 训练默认读取 Parquet。
- JSON 路径作为 fallback。

---

## Task P2.2: Feature Allowlist

### 目标

长期避免新字段自动进入模型。

### 新增文件

```text
config/feature_sets/vpa_d_sequence_pv_only.yaml
tests/test_feature_allowlist.py
```

### 示例

```yaml
feature_set_id: vpa_d_sequence
schema_version: v2_pv_only_001

include_patterns:
  - "ret_*"
  - "range_*"
  - "body_*"
  - "volume_ratio_*"
  - "amount_ratio_*"
  - "vpa_*_score"
  - "sequence_*_count"
  - "supply_score_*"
  - "demand_score_*"
  - "bull_bear_score_*"

exclude_columns:
  - industry_code
  - industry_name
  - industry_unknown
  - exchange
  - board
  - is_bse
  - is_st
  - is_paused
  - can_buy_next_open
  - can_sell_next_open
```

### 达标标准

- 训练特征由 allowlist 决定。
- 新增 feature 不会自动进入模型。
- 行业/交易/宇宙字段永远不会进入训练矩阵。
- feature_schema 可版本化。

---

## Task P2.3: Daily Signal 默认使用 Parquet Feature Store

### 目标

daily signal 不再解析 `features_json`。

### 修改文件

```text
ml_stock_selector/serving/daily_signal.py
tests/test_daily_signal_feature_store.py
```

### 流程

```text
as_of_date Parquet features
        ↓
join ml_tradeability_daily
        ↓
exclude_bse
        ↓
active three models
        ↓
prediction raw
        ↓
SQL rank_pct / trade_score_v2
        ↓
portfolio v2
```

### 达标标准

- daily signal 不读取 `features_json`。
- daily signal 读取 as_of_date 对应 Parquet。
- daily signal join tradeability metadata。
- BSE / ST / paused / can_buy 约束生效。
- 输出 target portfolio 和 blocked reasons。

---

## Task P2.4: Legacy Fallback and Deprecation Notice

### 目标

保留兼容路径，降低破坏性。

### 规则

1. 如果 Feature Store 不存在，可以 fallback 到旧 `features_json` 路径。
2. fallback 必须打印 warning：
   ```text
   features_json training path is legacy and not recommended for production-scale walk-forward
   ```
3. 文档中明确：
   ```text
   production-scale training must use Parquet Feature Store
   ```
4. 正式 walk-forward 默认拒绝 JSON 全量路径，除非显式：
   ```text
   --allow-legacy-json-path
   ```

### 达标标准

- 小规模测试仍可使用 JSON fallback。
- 正式 walk-forward 默认使用 Parquet Feature Store。
- 文档更新。

---

# 5. 推荐执行顺序

## P0 执行顺序

```text
1. P0.1 export_ml_feature_store.py
2. P0.2 feature_store_reader.py
3. P0.3 run_ml_walkforward.py --fold-id / --use-feature-store
4. P0.4 matrix_cache.py
5. P0.5 三模型从 Fold Matrix Cache 训练
6. P0.6 分块预测 + SQL rank_pct/trade_score_v2
7. P0.7 wf_2020 单 fold 验收
```

## P1 执行顺序

```text
1. P1.1 Feature Store 版本化
2. P1.2 Fold manifest / resume
3. P1.3 full multi-fold walk-forward
4. P1.4 backtest candidate join + metrics
```

## P2 执行顺序

```text
1. P2.1 feature_mart 直接写 Parquet
2. P2.2 feature allowlist
3. P2.3 daily signal 使用 Parquet
4. P2.4 legacy fallback / deprecation notice
```

---

# 6. 32G 机器推荐参数

```text
feature export chunk_size: 20,000
Parquet row_group_size: 50,000
compression: zstd
dtype: float32

matrix batch_size: 50,000
prediction chunk_size: 50,000

LightGBM:
  num_threads = 8
  force_col_wise = true
  max_bin = 63
  num_leaves = 31
  min_data_in_leaf = 500
  feature_fraction = 0.8
  bagging_fraction = 0.8
  bagging_freq = 1
```

如果仍然 OOM：

```text
1. chunk_size 降到 10,000
2. prediction chunk_size 降到 20,000
3. num_leaves 降到 15
4. max_bin 保持 63
5. 只跑 wf_2020
6. 暂时只导出 numeric features
```

---

# 7. Definition of Done

本实施计划完成的最终标准：

```text
1. features_json 不再作为正式 walk-forward 的训练输入。
2. Parquet Feature Store 可从 2015-01-05 覆盖到 2026-05-29。
3. Feature Store 不包含行业/交易/宇宙 metadata。
4. run_ml_walkforward.py 支持 --fold-id。
5. wf_2020 可在 32G 机器上跑通。
6. 三模型共享 Fold Matrix Cache。
7. 三模型训练不读取 ml_feature_mart_daily。
8. 三模型训练不解析 features_json。
9. test 预测分块执行。
10. rank_pct 和 trade_score_v2 由 DuckDB SQL 生成。
11. 北交所不进入训练、预测、组合、回测、日信号。
12. full multi-fold walk-forward 可恢复。
13. ml_backtest_metrics 按 fold/run 落库。
14. daily signal 默认读取 Parquet Feature Store。
15. legacy JSON 路径仅作为小规模 fallback。
```

---

# 8. Commit Discipline

每个 task 单独 commit。

Commit message 格式：

```text
<imperative summary>

<why this task exists>

Constraint: production-scale three-model training must use Parquet feature store, exclude BSE in VPA-ML, and keep industry/tradeability metadata out of model features
Confidence: <high|medium|low>
Scope-risk: <narrow|moderate|broad>
Tested: <exact pytest command>
```

示例：

```bash
git commit -m "Export VPA-ML features to partitioned Parquet store" -m "The walk-forward runner cannot train on production-scale features_json through pandas without OOM on a 32G machine. This change adds a chunked exporter that materializes model features as a partitioned Parquet feature store for fold-level training.

Constraint: production-scale three-model training must use Parquet feature store, exclude BSE in VPA-ML, and keep industry/tradeability metadata out of model features
Confidence: high
Scope-risk: moderate
Tested: python -m pytest tests/test_feature_store_export.py tests/test_feature_store_schema.py -v"
```
