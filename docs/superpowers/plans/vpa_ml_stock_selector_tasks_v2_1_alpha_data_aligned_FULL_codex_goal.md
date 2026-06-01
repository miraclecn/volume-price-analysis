# VPA-Enhanced ML Stock Selector Implementation Plan v2.1-alpha-data-aligned-full

> **For Codex `/goal` execution in `miraclecn/volume-price-analysis`**
>
> This document is the full expanded VPA-ML implementation plan aligned with `alpha-data`.
>
> It is **not** only a data-preparation document.  
> The data-preparation responsibilities move to `alpha-data`, while this document still covers the complete downstream ML stock-selection system:
>
> - alpha-data contract validation;
> - VPA schema validation;
> - OHLCV baseline features from normalized bars;
> - ML tradeability mart;
> - ML feature mart;
> - future labels;
> - training samples;
> - feature matrix encoding;
> - LightGBM ranker / regressor / risk models;
> - batch predictions;
> - scoring;
> - 10-15 stock portfolio construction;
> - T+1 execution backtest;
> - metrics;
> - walk-forward experiment runner;
> - daily signal generation;
> - model registry and split readiness.

---

## 0. Executive Goal

Build a low-coupling `ml_stock_selector` subsystem under `volume-price-analysis`.

The upstream data chain is:

```text
alpha-data
  research_source.duckdb
  stock_bar_normalized_daily
        ↓
volume-price-analysis
  vpa_structure_recognizer -> vpa_* tables
        ↓
ml_stock_selector
  ml_tradeability_daily
  ml_feature_mart_daily
  ml_labels_daily
  model artifacts
  ml_predictions_daily
  ml_portfolio_targets_daily
  ml_backtest_*
```

The business goal is:

```text
Use objective multi-window VPA labels and OHLCV-derived features
to rank the full market cross-section,
select 10-15 stocks,
and validate the strategy with leakage-safe T+1 execution.
```

---

## 1. Responsibility Boundary

### 1.1 alpha-data Owns

`alpha-data` owns source data preparation:

```text
raw ingestion
PIT daily bars
adjusted OHLC fields
ST flag
pause flag
limit-up / limit-down fields
trade calendar
industry PIT/reference fields
stock_bar_normalized_daily
source-data quality reports
```

### 1.2 VPA Owns

`vpa_structure_recognizer` owns deterministic VPA derivation:

```text
vpa_features
vpa_trend_context
vpa_bar_context_labels
vpa_sequence_stats
vpa_structure_state
top-down VPA reports
```

### 1.3 VPA-ML Owns

`ml_stock_selector` owns all downstream ML and strategy artifacts:

```text
alpha-data contract check
VPA schema contract check
OHLCV baseline features from normalized bars
ml_tradeability_daily with ADV and next-open fields
ml_feature_mart_daily
ml_labels_daily
training samples
feature matrix schemas
model artifacts
model registry
batch predictions
trade scoring
portfolio targets
backtest orders / positions / NAV / metrics
walk-forward experiments
daily signals
```

### 1.4 Explicitly Not Owned by VPA-ML

VPA-ML must not duplicate source-level joins owned by `alpha-data`:

```text
daily_bar_pit direct joins
tradeability_state_daily direct joins
industry_classification_pit direct joins
raw market data cleaning
price adjustment logic
canonical source bar construction
```

VPA-ML must read:

```text
stock_bar_normalized_daily
```

or a DataFrame with the same contract.

---

## 2. Required Upstream Contract

### 2.1 `stock_bar_normalized_daily`

Required columns:

```text
trade_date
code
open
high
low
close
prev_close
volume
amount
turnover_rate
is_st
is_paused
limit_up
limit_down
industry_code
industry_name
```

Rules:

- `open/high/low/close/prev_close` must be consistently adjusted.
- `is_st`, `is_paused`, `limit_up`, `limit_down` must reflect the source data's best known tradeability state.
- `industry_code` is the machine grouping key.
- `industry_name` is used for readable reports; if missing, warn but do not fail unless portfolio reports require it.

---

## 3. Directory Structure

Create:

```text
ml_stock_selector/
  __init__.py
  constants.py
  config.py
  storage.py
  contracts/
    __init__.py
    alpha_data_contract.py
    vpa_schema.py
    ml_schema.py
  data_access.py
  ohlcv_features.py
  tradeability.py
  feature_mart.py
  label_builder.py
  sample_builder.py
  feature_matrix.py
  datasets.py
  prediction.py
  scoring.py
  registry.py
  models/
    __init__.py
    artifacts.py
    alpha_ranker.py
    alpha_regressor.py
    risk_model.py
    calibrator.py
  portfolio/
    __init__.py
    constraints.py
    constructor.py
    allocator.py
  backtest/
    __init__.py
    execution.py
    engine.py
    metrics.py
    reports.py
    walkforward.py
  serving/
    __init__.py
    artifact_loader.py
    daily_signal.py
```

Scripts:

```text
scripts/run_alpha_data_contract_check.py
scripts/run_ml_schema_check.py
scripts/run_ml_feature_mart.py
scripts/build_ml_labels.py
scripts/train_ml_models.py
scripts/run_ml_batch_predict.py
scripts/run_ml_backtest.py
scripts/run_ml_walkforward.py
scripts/run_ml_daily_signal.py
```

Configs:

```text
config/ml_default.toml
config/ml_vpa.toml
config/ml_backtest.toml
```

SQL:

```text
sql/create_ml_tables.sql
```

---

## 4. Shared Constants

Create `ml_stock_selector/constants.py`.

```python
DEFAULT_FEATURE_WINDOWS = [5, 10, 20, 60, 120, 240]
DEFAULT_HORIZONS = [1, 5, 10]

FEATURE_SET_BASELINE_A = "baseline_a_ohlcv"
FEATURE_SET_BASELINE_B = "baseline_b_vpa_numeric"
FEATURE_SET_VPA_C = "vpa_c_bar_context"
FEATURE_SET_VPA_D = "vpa_d_sequence"
FEATURE_SET_VPA_E = "vpa_e_structure_state"

MODEL_TYPE_RANKER = "alpha_ranker"
MODEL_TYPE_REGRESSOR = "alpha_regressor"
MODEL_TYPE_RISK = "risk_model"

EXECUTION_NEXT_OPEN = "next_open"
EXECUTION_NEXT_VWAP = "next_vwap"

LABEL_BASE_FROM_CLOSE = "from_close"
LABEL_BASE_FROM_NEXT_OPEN = "from_next_open"

DEFAULT_MAIN_HORIZON = 5
DEFAULT_TARGET_POSITIONS = 12
DEFAULT_HARD_MAX_POSITIONS = 15

UNKNOWN_CATEGORY = "__UNKNOWN__"
MISSING_CATEGORY = "__MISSING__"
```

---

## 5. ML Table Contracts

### 5.1 `ml_tradeability_daily`

Derived from alpha-data normalized bars.

```text
trade_date
code
industry_code
industry_name
is_st
is_paused
limit_up
limit_down
open
high
low
close
prev_close
amount
turnover_rate
adv20_amount
next_trade_date
next_open
next_limit_up
next_limit_down
next_is_paused
can_buy_next_open
can_sell_next_open
generated_at
```

Primary key:

```text
(trade_date, code)
```

### 5.2 `ml_feature_mart_daily`

```text
trade_date
code
feature_set_id
vpa_data_version
generated_at
industry_code
industry_name
is_st
is_paused
limit_up
limit_down
adv20_amount
can_buy_next_open
can_sell_next_open
features_json
```

Primary key:

```text
(trade_date, code, feature_set_id)
```

### 5.3 `ml_labels_daily`

Long format:

```text
trade_date
code
horizon_d
label_base
base_price
future_ret
future_max_gain
future_max_drawdown
future_score
future_rank_pct
rank_label
risk_label
outperform_market
generated_at
```

Primary key:

```text
(trade_date, code, horizon_d, label_base)
```

### 5.4 `ml_model_registry`

```text
model_id
model_type
feature_set_id
label_name
label_base
horizon_d
train_start
train_end
valid_start
valid_end
test_start
test_end
params_json
metrics_json
feature_schema_uri
artifact_uri
is_active
activated_at
deactivated_at
created_at
notes
```

Primary key:

```text
(model_id)
```

Active model uniqueness is enforced by code:

```text
(model_type, feature_set_id, label_name, label_base, horizon_d)
```

### 5.5 `ml_predictions_daily`

```text
trade_date
code
model_id
horizon_d
alpha_score
alpha_rank_pct
reg_score
risk_score
risk_rank_pct
context_score
liquidity_score
relative_strength_pct
resonance_pct
penalty_score
trade_score
feature_set_id
generated_at
```

Primary key:

```text
(trade_date, code, model_id, horizon_d)
```

### 5.6 `ml_portfolio_targets_daily`

```text
trade_date
portfolio_id
code
target_weight
rank_n
trade_score
entry_reason
generated_at
```

Primary key:

```text
(trade_date, portfolio_id, code)
```

### 5.7 Backtest Tables

```text
ml_backtest_orders
ml_backtest_positions
ml_backtest_nav
ml_backtest_metrics
```

---

## 6. Feature Set Contract

| Feature Set | Includes | Excludes |
|---|---|---|
| `baseline_a_ohlcv` | OHLCV-derived numeric features only | all VPA features/labels/states |
| `baseline_b_vpa_numeric` | baseline A + `vpa_features` numeric columns | bar context labels, sequence labels, structure state |
| `vpa_c_bar_context` | baseline B + objective `vpa_bar_context_labels` | sequence, structure state |
| `vpa_d_sequence` | VPA C + `vpa_sequence_stats` | high-level structure state |
| `vpa_e_structure_state` | VPA D + selected `vpa_structure_state` fields | textual explanation fields unless explicitly enabled |

Default feature set:

```text
vpa_d_sequence
```

`final_state`, `final_rating`, `confidence`, and top-down scores are allowed only in `vpa_e_structure_state`.

---

## 7. Ranking Label Contract

Default `rank_label` must emphasize the top of the cross-section.

```python
def rank_label_from_pct(rank_pct: float) -> int:
    if rank_pct >= 0.99:
        return 4
    if rank_pct >= 0.95:
        return 3
    if rank_pct >= 0.90:
        return 2
    if rank_pct >= 0.70:
        return 1
    return 0
```

Default production `label_base`:

```text
from_next_open
```

`from_close` is allowed for research comparison.

---

# Phase 1: Contract-First Data Foundation

## Phase 1 Objective

Build:

```text
alpha-data normalized bars
        ↓
alpha-data contract check
        ↓
OHLCV baseline features
        ↓
ml_tradeability_daily
        ↓
VPA schema check
        ↓
ml_feature_mart_daily
        ↓
ml_labels_daily
        ↓
training samples
        ↓
feature matrix
        ↓
minimum ranker
        ↓
batch predictions
```

## Phase 1 Exit Criteria

```bash
python -m pytest \
  tests/test_alpha_data_contract_for_ml.py \
  tests/test_ml_storage_schema.py \
  tests/test_vpa_schema_contract.py \
  tests/test_ohlcv_features.py \
  tests/test_ml_tradeability.py \
  tests/test_ml_feature_mart.py \
  tests/test_ml_label_builder.py \
  tests/test_sample_builder.py \
  tests/test_feature_matrix.py \
  tests/test_ml_datasets.py \
  tests/test_alpha_ranker.py \
  tests/test_ml_prediction.py \
  tests/test_ml_pipeline_smoke.py \
  -v
```

---

## Milestone 1.1: Config and Package Skeleton

### Task 1.1.1: Add ML Package Skeleton

**Files**

Create:

```text
ml_stock_selector/__init__.py
ml_stock_selector/constants.py
ml_stock_selector/config.py
config/ml_default.toml
config/ml_vpa.toml
config/ml_backtest.toml
tests/test_ml_config.py
```

Modify:

```text
pyproject.toml
```

**Interfaces**

```python
@dataclass(frozen=True)
class MLConfig:
    data: dict[str, object]
    features: dict[str, object]
    labels: dict[str, object]
    split: dict[str, object]
    model: dict[str, object]
    portfolio: dict[str, object]
    backtest: dict[str, object]

def load_ml_config(path: Path | str) -> MLConfig:
    ...
```

**Config Requirements**

`ml_default.toml` must include:

```toml
[data]
alpha_data_db = "outputs/research_source.duckdb"
vpa_db = "outputs/vpa.duckdb"
ml_db = "outputs/ml/ml.duckdb"
normalized_bars_table = "stock_bar_normalized_daily"
artifact_dir = "outputs/ml/artifacts"
report_dir = "outputs/ml/reports"

[features]
windows = [5, 10, 20, 60, 120, 240]
feature_set_id = "vpa_d_sequence"
include_structure_state = false

[labels]
horizons = [1, 5, 10]
main_horizon = 5
label_base = "from_next_open"
risk_drawdown_threshold = -0.05

[split]
embargo_days = 10
folds = [
  { train_start = "2018-01-01", train_end = "2020-12-31", valid_start = "2021-01-01", valid_end = "2021-12-31", test_start = "2022-01-01", test_end = "2022-12-31" }
]

[model.alpha_ranker]
objective = "lambdarank"
metric = "ndcg"
eval_at = [10, 15]
lambdarank_truncation_level = 18
num_leaves = 63
learning_rate = 0.05
feature_fraction = 0.8
bagging_fraction = 0.8
bagging_freq = 1
min_data_in_leaf = 300
lambda_l2 = 5.0

[portfolio]
target_positions = 12
hard_max_positions = 15
max_industry_names = 3
max_new_entries_per_day = 4
single_name_min_weight = 0.05
single_name_max_weight = 0.10
allow_cash = true
min_trade_score = 0.80

[backtest]
initial_cash = 1000000
execution_price = "next_open"
slippage_bps = 5
commission_bps = 3
stamp_duty_bps = 5
a_share_lot_size = 100
allow_fractional_shares = true
```

**Acceptance Criteria**

- Config loads.
- Contradictory share-accounting settings are rejected.
- Tests pass.

---

## Milestone 1.2: Storage

### Task 1.2.1: Create ML Tables and Upsert Helpers

**Files**

Create:

```text
sql/create_ml_tables.sql
ml_stock_selector/storage.py
tests/test_ml_storage_schema.py
```

**Interfaces**

```python
def init_ml_db(path: Path | str) -> duckdb.DuckDBPyConnection:
    ...

def upsert_dataframe(
    con: duckdb.DuckDBPyConnection,
    table_name: str,
    frame: pd.DataFrame,
    key_columns: list[str],
) -> None:
    ...
```

**Acceptance Criteria**

- All `ml_*` tables exist.
- `ml_labels_daily` is long-format.
- Upsert is idempotent.
- Tests pass.

---

## Milestone 1.3: alpha-data Contract

### Task 1.3.1: Validate alpha-data Normalized Bar Contract

**Files**

Create:

```text
ml_stock_selector/contracts/alpha_data_contract.py
scripts/run_alpha_data_contract_check.py
tests/test_alpha_data_contract_for_ml.py
```

**Interfaces**

```python
REQUIRED_NORMALIZED_BAR_COLUMNS = {
    "trade_date",
    "code",
    "open",
    "high",
    "low",
    "close",
    "prev_close",
    "volume",
    "amount",
    "turnover_rate",
    "is_st",
    "is_paused",
    "limit_up",
    "limit_down",
    "industry_code",
    "industry_name",
}

@dataclass(frozen=True)
class AlphaDataContractResult:
    ok: bool
    missing_tables: list[str]
    missing_columns: dict[str, list[str]]
    warnings: list[str]

def validate_alpha_data_contract(
    con: duckdb.DuckDBPyConnection,
    normalized_table: str = "stock_bar_normalized_daily",
) -> AlphaDataContractResult:
    ...

def assert_alpha_data_contract(
    con: duckdb.DuckDBPyConnection,
    normalized_table: str = "stock_bar_normalized_daily",
) -> None:
    ...
```

**Acceptance Criteria**

- Missing normalized table fails.
- Missing core price/tradeability columns fails.
- Missing `industry_name` warns if `industry_code` exists.
- CLI exits non-zero on hard failure.
- Tests pass.

---

## Milestone 1.4: Data Access

### Task 1.4.1: Load Normalized Stock Bars

**Files**

Create:

```text
ml_stock_selector/data_access.py
tests/test_alpha_data_contract_for_ml.py
```

**Interfaces**

```python
def load_normalized_stock_bars(
    alpha_data_db_path: str,
    start_date: str,
    end_date: str,
    table_name: str = "stock_bar_normalized_daily",
) -> pd.DataFrame:
    ...
```

**Rules**

- Read only `stock_bar_normalized_daily`.
- Do not join `daily_bar_pit`, `tradeability_state_daily`, or `industry_classification_pit`.
- Sort by `(code, trade_date)`.
- Return normalized contract columns.

**Acceptance Criteria**

- Fixture loads.
- Missing table fails via contract checker.
- Tests pass.

---

## Milestone 1.5: VPA Schema Contract

### Task 1.5.1: Validate `vpa_*` Contracts

**Files**

Create:

```text
ml_stock_selector/contracts/vpa_schema.py
scripts/run_ml_schema_check.py
tests/test_vpa_schema_contract.py
```

**Required Tables**

```text
vpa_features
vpa_bar_context_labels
vpa_sequence_stats
vpa_structure_state
```

**Acceptance Criteria**

- Missing VPA table errors clearly.
- Optional column alias mapping works.
- Schema snapshot writes valid JSON.
- Tests pass.

---

## Milestone 1.6: OHLCV Baseline Features

### Task 1.6.1: Build Baseline OHLCV Features

**Files**

Create:

```text
ml_stock_selector/ohlcv_features.py
tests/test_ohlcv_features.py
```

**Interfaces**

```python
def build_ohlcv_features(
    normalized_bars: pd.DataFrame,
    windows: list[int],
) -> pd.DataFrame:
    ...
```

**Required Features**

```text
ret_1d
open_gap_pct
range_pct
body_pct
upper_shadow_pct
lower_shadow_pct
close_position
amount
turnover_rate
```

For each window:

```text
ret_{n}d
volatility_{n}d
amount_ratio_{n}d
volume_ratio_{n}d
turnover_mean_{n}d
high_distance_{n}d
low_distance_{n}d
```

**Rules**

- Use only bars up to T.
- Do not read raw alpha-data source tables.

**Acceptance Criteria**

- Hand-calculated fixture values match.
- No future data is used.
- Tests pass.

---

## Milestone 1.7: Tradeability Mart

### Task 1.7.1: Build `ml_tradeability_daily`

**Files**

Create:

```text
ml_stock_selector/tradeability.py
tests/test_ml_tradeability.py
```

**Interfaces**

```python
def build_tradeability_mart(
    normalized_bars: pd.DataFrame,
    adv_window: int = 20,
) -> pd.DataFrame:
    ...
```

**Rules**

- `adv20_amount` uses T and prior bars only.
- `next_*` fields use next row per stock.
- `can_buy_next_open = not next_is_paused and next_open < next_limit_up`.
- `can_sell_next_open = not next_is_paused and next_open > next_limit_down`.

**Acceptance Criteria**

- Paused next day rejects.
- Limit-up next open rejects buy.
- Limit-down next open rejects sell.
- No next bar yields false flags.
- Tests pass.

---

## Milestone 1.8: Feature Mart

### Task 1.8.1: Pivot VPA Numeric Features

**Files**

Create/modify:

```text
ml_stock_selector/feature_mart.py
tests/test_ml_feature_mart.py
```

**Interfaces**

```python
def build_vpa_numeric_features(
    vpa_db_path: str,
    start_date: str,
    end_date: str,
    windows: list[int],
) -> pd.DataFrame:
    ...
```

**Rules**

- Read `vpa_features`.
- `scope_type = 'stock'`.
- Pivot by `window_n`.
- Use column aliases.

**Acceptance Criteria**

- One row per `(trade_date, code)`.
- Window suffix columns appear.
- Tests pass.

---

### Task 1.8.2: Add Bar Context and Sequence Features

**Interfaces**

```python
def build_vpa_bar_context_features(...) -> pd.DataFrame:
    ...

def build_vpa_sequence_features(...) -> pd.DataFrame:
    ...
```

**Rules**

- Add `raw_label`, `bull_bear_score`, `supply_score`, `demand_score`, `volatility_score`.
- Add sequence fields when available.
- Do not include state fields.

**Acceptance Criteria**

- Objective VPA labels appear.
- Missing optional sequence fields do not crash.
- Tests pass.

---

### Task 1.8.3: Add Optional Structure State for VPA E Only

**Interfaces**

```python
def build_structure_state_features(...) -> pd.DataFrame:
    ...

def apply_feature_set_filter(
    features: pd.DataFrame,
    feature_set_id: str,
) -> pd.DataFrame:
    ...
```

**Rules**

- `final_state`, `final_rating`, and high-level scores appear only in `vpa_e_structure_state`.
- VPA D excludes them.

**Acceptance Criteria**

- Feature set tests pass.

---

### Task 1.8.4: Assemble `ml_feature_mart_daily`

**Interfaces**

```python
def build_feature_mart(
    vpa_db_path: str,
    normalized_bars: pd.DataFrame,
    start_date: str,
    end_date: str,
    feature_set_id: str,
    windows: list[int],
    tradeability: pd.DataFrame,
) -> pd.DataFrame:
    ...
```

**Rules**

- Always include OHLCV baseline features.
- Add VPA layers by feature set.
- Join tradeability on `(trade_date, code)`.
- Do not join raw alpha-data tables.
- Deterministic sorted-key `features_json`.

**Acceptance Criteria**

- Baseline A has no VPA fields.
- VPA D has OHLCV + VPA numeric + bar context + sequence.
- CLI writes to `ml_feature_mart_daily`.
- Tests pass.

---

## Milestone 1.9: Labels

### Task 1.9.1: Build Long-Format Future Labels

**Files**

Create:

```text
ml_stock_selector/label_builder.py
scripts/build_ml_labels.py
tests/test_ml_label_builder.py
```

**Interfaces**

```python
def build_labels(
    normalized_bars: pd.DataFrame,
    horizons: list[int],
    risk_drawdown_threshold: float = -0.05,
    label_bases: list[str] = ["from_close", "from_next_open"],
) -> pd.DataFrame:
    ...
```

**Rules**

For `from_close`:

```text
base_price = T close
future window = T+1 ... T+h
```

For `from_next_open`:

```text
base_price = T+1 open
future window = T+1 ... T+h
```

Output long format.

**Acceptance Criteria**

- No T-day path leakage.
- Incomplete horizons dropped.
- Head-heavy rank labels generated.
- Tests pass.

---

## Milestone 1.10: Training Samples and Matrix

### Task 1.10.1: Build Training Samples

**Files**

Create:

```text
ml_stock_selector/sample_builder.py
tests/test_sample_builder.py
```

**Interfaces**

```python
def build_training_samples(
    feature_mart: pd.DataFrame,
    labels: pd.DataFrame,
    feature_set_id: str,
    horizon_d: int,
    label_base: str,
) -> pd.DataFrame:
    ...
```

**Acceptance Criteria**

- Joins on `(trade_date, code)`.
- Filters horizon and label base.
- Drops invalid labels.
- Tests pass.

---

### Task 1.10.2: Build Feature Matrix

**Files**

Create:

```text
ml_stock_selector/feature_matrix.py
tests/test_feature_matrix.py
```

**Interfaces**

```python
@dataclass(frozen=True)
class FeatureSchema:
    feature_set_id: str
    numeric_columns: list[str]
    categorical_columns: list[str]
    output_columns: list[str]
    category_levels: dict[str, list[str]]
    fill_values: dict[str, object]
    schema_version: str

def build_feature_matrix(
    feature_mart_or_samples: pd.DataFrame,
    feature_set_id: str,
    schema: FeatureSchema | None = None,
    fit: bool = False,
) -> tuple[pd.DataFrame, FeatureSchema]:
    ...
```

**Rules**

- Fit schema only on training data.
- Use one-hot encoding v1.
- Unknown categories become `__UNKNOWN__`.
- Missing categories become `__MISSING__`.
- Missing numeric values default to `0.0`.

**Acceptance Criteria**

- Training and inference columns align.
- Schema roundtrip works.
- Tests pass.

---

## Milestone 1.11: Dataset, Ranker, Prediction

### Task 1.11.1: Walk-Forward Dataset Split

**Files**

Create:

```text
ml_stock_selector/datasets.py
tests/test_ml_datasets.py
```

**Interfaces**

```python
@dataclass(frozen=True)
class DateRange:
    start: str
    end: str

@dataclass(frozen=True)
class TrainValidTestSplit:
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame

def make_walk_forward_split(...):
    ...

def build_lgbm_group(frame: pd.DataFrame) -> list[int]:
    ...
```

**Acceptance Criteria**

- Embargo works.
- Group sizes sum to sample count.
- Tests pass.

---

### Task 1.11.2: Train Minimum Alpha Ranker

**Files**

Create:

```text
ml_stock_selector/models/artifacts.py
ml_stock_selector/models/alpha_ranker.py
scripts/train_ml_models.py
tests/test_alpha_ranker.py
```

**Interfaces**

```python
@dataclass(frozen=True)
class ModelArtifact:
    model_id: str
    model_type: str
    feature_set_id: str
    label_name: str
    label_base: str
    horizon_d: int
    feature_schema_uri: Path
    artifact_uri: Path
    artifact_dir: Path
    metrics: dict[str, float]

def train_alpha_ranker(...) -> ModelArtifact:
    ...
```

**Acceptance Criteria**

- Trains on synthetic panel.
- Saves model + schema.
- Reload predicts.
- Tests pass.

---

### Task 1.11.3: Batch Predict

**Files**

Create:

```text
ml_stock_selector/prediction.py
scripts/run_ml_batch_predict.py
tests/test_ml_prediction.py
```

**Interfaces**

```python
def predict_with_model(
    feature_mart: pd.DataFrame,
    artifact: ModelArtifact,
) -> pd.DataFrame:
    ...

def build_prediction_rows(...):
    ...

def upsert_predictions(...):
    ...
```

**Rules**

- Use saved feature schema.
- Preserve `trade_date`, `code`.
- Fill `alpha_score`.
- Do not compute `trade_score`.

**Acceptance Criteria**

- Upsert works.
- Feature mismatch errors clearly.
- Tests pass.

---

# Phase 2: Portfolio and Backtest

## Phase 2 Objective

Convert predictions into executable target portfolios and simulate T+1 execution.

## Phase 2 Exit Criteria

```bash
python -m pytest \
  tests/test_ml_scoring.py \
  tests/test_portfolio_constructor.py \
  tests/test_backtest_execution.py \
  tests/test_backtest_engine.py \
  tests/test_backtest_metrics.py \
  -v
```

---

## Milestone 2.1: Score Sources

### Task 2.1.1: Add Context and Liquidity Scores

**Files**

Create/modify:

```text
ml_stock_selector/scoring.py
tests/test_ml_scoring.py
```

**Interfaces**

```python
def add_context_score(candidates: pd.DataFrame) -> pd.DataFrame:
    ...

def add_liquidity_score(candidates: pd.DataFrame) -> pd.DataFrame:
    ...
```

**Rules**

Context score sources:

```text
confidence
self_score
sector_score
market_score
resonance_score
```

Liquidity score sources:

```text
adv20_amount
amount
turnover_rate
```

Defaults:

```text
context_score = 0.5 if unavailable
liquidity_score = 0.5 if unavailable
```

**Acceptance Criteria**

- Missing fields default.
- Existing fields convert to daily percentiles.
- Tests pass.

---

## Milestone 2.2: Trade Score

### Task 2.2.1: Compute Trade Score

**Interfaces**

```python
def score_candidates(predictions: pd.DataFrame) -> pd.DataFrame:
    ...
```

Formula:

```text
trade_score =
0.60 * alpha_rank_pct
+ 0.15 * context_score_pct
+ 0.10 * liquidity_score_pct
+ 0.05 * relative_strength_pct
+ 0.10 * resonance_pct
- 0.30 * risk_rank_pct
- penalty_score
```

**Acceptance Criteria**

- Higher alpha increases score.
- Higher risk decreases score.
- Stable tie order.
- Tests pass.

---

## Milestone 2.3: Portfolio Construction

### Task 2.3.1: Hard Filters

**Files**

Create:

```text
ml_stock_selector/portfolio/constraints.py
tests/test_portfolio_constructor.py
```

**Interfaces**

```python
@dataclass(frozen=True)
class PortfolioConstraints:
    target_positions: int = 12
    hard_max_positions: int = 15
    max_industry_names: int = 3
    max_new_entries_per_day: int = 4
    min_adv20_amount: float | None = None
    min_trade_score: float = 0.80
    allow_cash: bool = True

def apply_hard_filters(
    candidates: pd.DataFrame,
    constraints: PortfolioConstraints,
) -> pd.DataFrame:
    ...
```

Exclude:

```text
is_st
is_paused
can_buy_next_open == false
adv20_amount below threshold
trade_score below threshold
```

**Acceptance Criteria**

- Hard filters tested.
- Empty output allowed.

---

### Task 2.3.2: Current-Holdings-Aware Selection

**Files**

Create:

```text
ml_stock_selector/portfolio/constructor.py
```

**Interfaces**

```python
def construct_portfolio_targets(
    scored_candidates: pd.DataFrame,
    constraints: PortfolioConstraints,
    portfolio_id: str,
    current_holdings: pd.DataFrame | None = None,
) -> pd.DataFrame:
    ...
```

**Rules**

- Enforce hard max.
- Enforce industry max.
- Enforce max new entries.
- Existing holdings may be retained.
- Removed holdings are sold by the backtest engine.

**Acceptance Criteria**

- New entries <= max.
- Industry limit holds.
- Fewer than target allowed.
- Tests pass.

---

### Task 2.3.3: Allocate Weights

**Files**

Create:

```text
ml_stock_selector/portfolio/allocator.py
```

**Interfaces**

```python
def allocate_weights(
    selected: pd.DataFrame,
    min_weight: float,
    max_weight: float,
    allow_cash: bool,
) -> pd.DataFrame:
    ...
```

**Acceptance Criteria**

- Weight sum <= 1 if cash allowed.
- Weight bounds respected.
- Tests pass.

---

## Milestone 2.4: Execution

### Task 2.4.1: Simulate T+1 Execution

**Files**

Create:

```text
ml_stock_selector/backtest/execution.py
tests/test_backtest_execution.py
```

**Interfaces**

```python
@dataclass(frozen=True)
class ExecutionConfig:
    execution_price: str = "next_open"
    slippage_bps: float = 5.0
    commission_bps: float = 3.0
    stamp_duty_bps: float = 5.0
    a_share_lot_size: int = 100
    allow_fractional_shares: bool = True

def simulate_rebalance_orders(...):
    ...
```

**Rules**

- `sim_date > decision_date`.
- Reject paused next day.
- Reject limit-up buy.
- Reject limit-down sell.
- Commission both sides.
- Stamp duty sell side only.
- Fractional/lot mode controlled by config.

**Acceptance Criteria**

- Leakage guard passes.
- Cost rules tested.
- Tests pass.

---

## Milestone 2.5: Backtest Engine

### Task 2.5.1: Backtest Orders, Positions, NAV

**Files**

Create:

```text
ml_stock_selector/backtest/engine.py
scripts/run_ml_backtest.py
tests/test_backtest_engine.py
```

**Interfaces**

```python
@dataclass(frozen=True)
class BacktestConfig:
    initial_cash: float
    portfolio_id: str
    execution: ExecutionConfig

@dataclass(frozen=True)
class BacktestResult:
    orders: pd.DataFrame
    positions: pd.DataFrame
    nav: pd.DataFrame

def run_backtest(...):
    ...
```

**Rules**

- Add target weight 0 for current holdings absent from new target list.
- Generate sell orders for removed holdings.
- Mark to close using normalized bars.
- Preserve cash if target weights sum below 1.

**Acceptance Criteria**

- Removed holdings are sold.
- NAV accounts for costs.
- Tests pass.

---

## Milestone 2.6: Metrics

### Task 2.6.1: Backtest Metrics and Reports

**Files**

Create:

```text
ml_stock_selector/backtest/metrics.py
ml_stock_selector/backtest/reports.py
tests/test_backtest_metrics.py
```

Metrics:

```text
RankIC
NDCG@10
NDCG@15
Top-N mean future return
annualized return
max drawdown
turnover
profit/loss ratio
yearly slices
```

**Acceptance Criteria**

- Fixture metrics match hand calculations.
- Reports write under `outputs/ml/reports`.
- Tests pass.

---

# Phase 3: Governance, Experiments, Walk-Forward

## Phase 3 Exit Criteria

```bash
python -m pytest \
  tests/test_alpha_regressor.py \
  tests/test_risk_model.py \
  tests/test_ml_registry.py \
  tests/test_ml_feature_sets.py \
  tests/test_walkforward.py \
  tests/test_ml_pipeline_smoke.py \
  -v
```

---

## Milestone 3.1: Auxiliary Models

### Task 3.1.1: Add Alpha Regressor and Risk Model

**Files**

Create:

```text
ml_stock_selector/models/alpha_regressor.py
ml_stock_selector/models/risk_model.py
tests/test_alpha_regressor.py
tests/test_risk_model.py
```

**Acceptance Criteria**

- Both use `FeatureSchema`.
- Both save/load.
- Both predict correct lengths.
- Tests pass.

---

## Milestone 3.2: Registry

### Task 3.2.1: Model Registry and Activation

**Files**

Create:

```text
ml_stock_selector/registry.py
tests/test_ml_registry.py
```

**Interfaces**

```python
def register_model(...):
    ...

def activate_model(...):
    ...

def get_active_model(...):
    ...
```

**Acceptance Criteria**

- Activating one model deactivates same-key models.
- Missing active model errors.
- Tests pass.

---

## Milestone 3.3: Calibration

### Task 3.3.1: Cross-Sectional Calibration

**Files**

Create:

```text
ml_stock_selector/models/calibrator.py
tests/test_ml_scoring.py
```

**Interfaces**

```python
def cross_sectional_percentile(...):
    ...
```

**Acceptance Criteria**

- Percentiles in `[0, 1]`.
- Grouped by date.
- Tests pass.

---

## Milestone 3.4: Feature Ablation

### Task 3.4.1: Make A/B/C/D/E Feature Sets Executable

**Acceptance Criteria**

- Baseline A only OHLCV.
- Baseline B adds VPA numeric.
- VPA C adds bar context.
- VPA D adds sequence.
- VPA E adds structure state.
- VPA D excludes `final_state` and `final_rating`.

---

## Milestone 3.5: Walk-Forward Runner

### Task 3.5.1: Implement Walk-Forward Experiment Runner

**Files**

Create:

```text
ml_stock_selector/backtest/walkforward.py
scripts/run_ml_walkforward.py
tests/test_walkforward.py
```

**Interfaces**

```python
@dataclass(frozen=True)
class WalkForwardFoldResult:
    fold_id: str
    model_ids: list[str]
    predictions: pd.DataFrame
    targets: pd.DataFrame
    backtest_result: BacktestResult
    metrics: dict[str, float]

def run_walkforward_experiment(
    config: MLConfig,
    normalized_bars: pd.DataFrame,
    feature_mart: pd.DataFrame,
    labels: pd.DataFrame,
    tradeability: pd.DataFrame,
) -> list[WalkForwardFoldResult]:
    ...
```

**Flow**

1. Build samples.
2. Split with embargo.
3. Train alpha ranker.
4. Optionally train risk model.
5. Batch predict.
6. Add context/liquidity.
7. Score candidates.
8. Construct portfolio targets.
9. Backtest.
10. Compute metrics.
11. Persist outputs.

**Acceptance Criteria**

- At least two synthetic folds run.
- Predictions, targets, NAV, metrics exist.
- No filled order violates T+1.
- Tests pass.

---

# Phase 4: Daily Signal and Split Readiness

## Phase 4 Exit Criteria

```bash
python -m pytest \
  tests/test_daily_signal.py \
  tests/test_ml_contracts.py \
  tests/test_ml_pipeline_smoke.py \
  -v
```

---

## Milestone 4.1: Artifact Loading

### Task 4.1.1: Load Active Models

**Files**

Create:

```text
ml_stock_selector/serving/artifact_loader.py
tests/test_daily_signal.py
```

**Interfaces**

```python
def load_active_model(...):
    ...
```

**Acceptance Criteria**

- Loads model + schema.
- Missing active model errors.
- No hardcoded artifact paths.

---

## Milestone 4.2: Daily Signal

### Task 4.2.1: Generate Daily Predictions and Targets

**Files**

Create:

```text
ml_stock_selector/serving/daily_signal.py
scripts/run_ml_daily_signal.py
tests/test_daily_signal.py
```

**Interfaces**

```python
def generate_daily_signal(
    con: duckdb.DuckDBPyConnection,
    as_of_date: str,
    feature_set_id: str,
    horizon_d: int,
    portfolio_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ...
```

**Flow**

1. Verify alpha-data contract if rebuilding features.
2. Load normalized bars.
3. Build/update tradeability.
4. Read VPA tables.
5. Build/update feature mart.
6. Load active models.
7. Build matrix using saved schema.
8. Predict.
9. Add context/liquidity.
10. Score.
11. Construct target portfolio.
12. Upsert predictions and targets.

**Acceptance Criteria**

- No training is invoked.
- Predictions written.
- Targets written.
- Target count <= hard max.
- Tests pass.

---

## Milestone 4.3: Guardrails

### Task 4.3.1: Contract and Leakage Guard Tests

**Files**

Create:

```text
tests/test_ml_contracts.py
```

**Tests**

- `vpa_structure_recognizer` does not import `ml_stock_selector`.
- `ml_stock_selector` does not join raw alpha-data source tables.
- Filled orders satisfy `sim_date > decision_date`.
- Backtest fails if tradeability is required but missing.
- Missing feature schema fails serving.

---

## Milestone 4.4: Documentation

### Task 4.4.1: README and Operating Notes

**Files**

Create:

```text
docs/ml_stock_selector_operating_notes.md
```

Modify:

```text
README.md
```

Include:

```text
architecture boundary
alpha-data dependency
command order
feature sets
label bases
T+1 execution rule
model registry
backtest assumptions
known limitations
```

---

## Milestone 4.5: Split Readiness

### Task 4.5.1: Write Split-Readiness Checklist

**Files**

Create:

```text
docs/superpowers/specs/ml-stock-selector-split-readiness.md
```

Checklist:

```text
1. alpha-data normalized bar contract stable.
2. vpa_* schema stable.
3. ml_feature_mart_daily and ml_predictions_daily stable across at least two walk-forward runs.
4. daily signal reads only normalized bars, vpa_* tables, and model artifacts.
5. no private VPA implementation imports in ML.
6. backtest and daily inference use the same prediction and target contracts.
7. artifact naming and registry stable.
```

---

# Phase-Level Verification Commands

```bash
# Phase 1
python -m pytest \
  tests/test_alpha_data_contract_for_ml.py \
  tests/test_ml_storage_schema.py \
  tests/test_vpa_schema_contract.py \
  tests/test_ohlcv_features.py \
  tests/test_ml_tradeability.py \
  tests/test_ml_feature_mart.py \
  tests/test_ml_label_builder.py \
  tests/test_sample_builder.py \
  tests/test_feature_matrix.py \
  tests/test_ml_datasets.py \
  tests/test_alpha_ranker.py \
  tests/test_ml_prediction.py \
  tests/test_ml_pipeline_smoke.py \
  -v

# Phase 2
python -m pytest \
  tests/test_ml_scoring.py \
  tests/test_portfolio_constructor.py \
  tests/test_backtest_execution.py \
  tests/test_backtest_engine.py \
  tests/test_backtest_metrics.py \
  -v

# Phase 3
python -m pytest \
  tests/test_alpha_regressor.py \
  tests/test_risk_model.py \
  tests/test_ml_registry.py \
  tests/test_ml_feature_sets.py \
  tests/test_walkforward.py \
  tests/test_ml_pipeline_smoke.py \
  -v

# Phase 4
python -m pytest \
  tests/test_daily_signal.py \
  tests/test_ml_contracts.py \
  tests/test_ml_pipeline_smoke.py \
  -v

# Final
python -m pytest tests -v
```

---

# Summary of Changes from Non-Aligned v2.1

Moved out of VPA-ML and into alpha-data:

```text
canonical normalized bar source construction
raw PIT daily bar joins
raw tradeability source joins
raw industry PIT/reference joins
source data quality audit
```

Kept in VPA-ML:

```text
OHLCV baseline features from normalized bars
ADV20
next-open tradeability fields
can_buy_next_open / can_sell_next_open
future labels
rank labels
feature mart
feature matrix
model training
predictions
portfolio construction
backtest
daily signal
walk-forward experiments
```

Key dependency:

```text
alpha-data must expose stock_bar_normalized_daily.
```
