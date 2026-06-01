# VPA-Enhanced ML Stock Selector Four-Phase Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a low-coupling `ml_stock_selector` subsystem that consumes existing `vpa_*` tables, trains VPA-informed ranking models, produces 10-15 stock portfolio targets, and validates them through a lightweight T+1 backtest loop.

**Architecture:** Keep `vpa_structure_recognizer` as the upstream deterministic volume-price structure generator. Add sibling modules under `ml_stock_selector/` that read `vpa_*` contracts, write only `ml_*` contracts, persist model artifacts, generate daily predictions, construct constrained portfolios, and run leakage-safe walk-forward backtests. Do not make VPA import ML, and do not let ML depend on upstream raw-source table names except through existing source adapters.

**Tech Stack:** Python 3.11+, DuckDB, pandas, numpy, LightGBM, scikit-learn, pytest. Optional future state stores can use PostgreSQL or SQLite, but first implementation should keep DuckDB/Parquet outputs under `outputs/`.

---

## Source Documents

This plan implements the recommended scheme from:

- `docs/superpowers/specs/构建量价分析增强型机器学习选股系统工程设计规范.md`
- `docs/superpowers/specs/量价分析增强型机器学习选股系统实施研究.md`

The key recommendation is to add a low-coupling downstream ML subsystem instead of rewriting the existing VPA rule pipeline. The primary model is a LightGBM cross-sectional ranker because the business problem is daily Top-10/15 selection, not single-name up/down classification. The backtest module is a new lightweight execution simulator because existing `backtest_validator.py` is a posterior validation label helper, not a portfolio execution engine.

## Non-Negotiable Boundaries

- `vpa_structure_recognizer/` continues to own deterministic VPA feature, label, sequence, state, top-down rating, validation, and Excel output logic.
- `ml_stock_selector/` reads `vpa_*` tables and writes `ml_*` tables. It must not write to upstream source DuckDB files.
- `ml_stock_selector/` may reuse public helpers such as storage patterns and source adapters, but it must not require VPA modules to import ML modules.
- Model supervision targets must be future returns, future max gain/drawdown, cross-sectional rank labels, and risk labels. Do not train models to reproduce `final_state` or `final_rating`.
- Trading simulation uses T-day close information to decide, then executes no earlier than T+1 open or T+1 VWAP. Same-day open fills are invalid for this design.
- First implementation should avoid new heavyweight backtest frameworks. The repo-specific lightweight backtest contract is the expected path.

## Target File Structure

Create:

- `ml_stock_selector/__init__.py`
- `ml_stock_selector/config.py`
- `ml_stock_selector/storage.py`
- `ml_stock_selector/feature_mart.py`
- `ml_stock_selector/label_builder.py`
- `ml_stock_selector/datasets.py`
- `ml_stock_selector/scoring.py`
- `ml_stock_selector/registry.py`
- `ml_stock_selector/models/__init__.py`
- `ml_stock_selector/models/artifacts.py`
- `ml_stock_selector/models/alpha_ranker.py`
- `ml_stock_selector/models/alpha_regressor.py`
- `ml_stock_selector/models/risk_model.py`
- `ml_stock_selector/models/calibrator.py`
- `ml_stock_selector/portfolio/__init__.py`
- `ml_stock_selector/portfolio/constraints.py`
- `ml_stock_selector/portfolio/constructor.py`
- `ml_stock_selector/portfolio/allocator.py`
- `ml_stock_selector/backtest/__init__.py`
- `ml_stock_selector/backtest/execution.py`
- `ml_stock_selector/backtest/engine.py`
- `ml_stock_selector/backtest/metrics.py`
- `ml_stock_selector/backtest/reports.py`
- `ml_stock_selector/backtest/walkforward.py`
- `ml_stock_selector/serving/__init__.py`
- `ml_stock_selector/serving/artifact_loader.py`
- `ml_stock_selector/serving/daily_signal.py`
- `config/ml_default.toml`
- `config/ml_backtest.toml`
- `config/ml_vpa.toml`
- `sql/create_ml_tables.sql`
- `scripts/run_ml_feature_mart.py`
- `scripts/build_ml_labels.py`
- `scripts/train_ml_models.py`
- `scripts/run_ml_backtest.py`
- `scripts/run_ml_daily_signal.py`
- `tests/test_ml_storage_schema.py`
- `tests/test_ml_feature_mart.py`
- `tests/test_ml_label_builder.py`
- `tests/test_ml_datasets.py`
- `tests/test_alpha_ranker.py`
- `tests/test_alpha_regressor.py`
- `tests/test_risk_model.py`
- `tests/test_ml_scoring.py`
- `tests/test_portfolio_constructor.py`
- `tests/test_backtest_execution.py`
- `tests/test_backtest_engine.py`
- `tests/test_backtest_metrics.py`
- `tests/test_daily_signal.py`
- `tests/test_ml_pipeline_smoke.py`

Modify:

- `pyproject.toml`
- `README.md`

Do not modify analytical rules in:

- `vpa_structure_recognizer/feature_engineering.py`
- `vpa_structure_recognizer/bar_labeler.py`
- `vpa_structure_recognizer/sequence_analyzer.py`
- `vpa_structure_recognizer/state_classifier.py`
- `vpa_structure_recognizer/top_down_ranker.py`

## Shared Contracts

Use these names consistently across implementation tasks.

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
```

Primary `ml_*` tables:

- `ml_feature_mart_daily`: one row per `trade_date, code, feature_set_id`.
- `ml_labels_daily`: one row per `trade_date, code, horizon_d`.
- `ml_model_registry`: one row per model artifact version.
- `ml_predictions_daily`: one row per `trade_date, code, model_id, horizon_d`.
- `ml_portfolio_targets_daily`: one row per target holding.
- `ml_backtest_orders`: one row per simulated order.
- `ml_backtest_positions`: one row per simulated holding snapshot.
- `ml_backtest_nav`: one row per simulated trading day.
- `ml_backtest_metrics`: one row per backtest run and metric set.

## Four-Phase Delivery Map

| Phase | Name | Primary Outcome | Included Tasks |
|---|---|---|---|
| Phase 1 | Contract-First Data Foundation | `ml_*` tables, feature mart, labels, dataset split, minimum ranker train/predict path | Tasks 1-7 |
| Phase 2 | Decision-First Portfolio And Backtest | `trade_score -> portfolio targets -> T+1 execution -> NAV` closed loop | Tasks 8-12 |
| Phase 3 | Governance, Experiments, Reports | artifact registry, calibration, ablation matrix, reports, smoke tests, docs | Tasks 13-16 |
| Phase 4 | Daily Signal And Split Readiness | daily inference, stable contracts, repo split checklist, operational guardrails | Tasks 17-20 |

---

## Phase 1: Contract-First Data Foundation

**Objective:** Create the ML subsystem boundary and prove the repo can generate a trainable wide table plus leakage-aware labels from existing VPA and price contracts.

**Exit Criteria:**

- `python -m pytest tests/test_ml_storage_schema.py tests/test_ml_feature_mart.py tests/test_ml_label_builder.py tests/test_ml_datasets.py tests/test_alpha_ranker.py -v` passes.
- `python scripts/run_ml_feature_mart.py --help`, `python scripts/build_ml_labels.py --help`, and `python scripts/train_ml_models.py --help` run without crashing.
- A small fixture dataset can build `ml_feature_mart_daily`, `ml_labels_daily`, train a ranker, save an artifact, reload it, and predict.

### Task 1: Add ML Subsystem Skeleton And Config

**Files:**

- Create: `ml_stock_selector/__init__.py`
- Create: `ml_stock_selector/config.py`
- Create: `config/ml_default.toml`
- Create: `config/ml_vpa.toml`
- Create: `scripts/run_ml_feature_mart.py`
- Create: `scripts/build_ml_labels.py`
- Create: `scripts/train_ml_models.py`
- Modify: `pyproject.toml`
- Test: `tests/test_config.py`

- [ ] Add project dependencies in `pyproject.toml`.

```toml
[project]
name = "vpa-structure-recognizer"
version = "0.1.0"
description = "A-share multi-level volume-price structure recognizer"
requires-python = ">=3.11"
dependencies = [
    "duckdb",
    "lightgbm",
    "numpy",
    "openpyxl",
    "pandas",
    "scikit-learn",
]

[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests"]
```

- [ ] Create `config/ml_default.toml`.

```toml
[data]
vpa_db = "outputs/vpa.duckdb"
ml_db = "outputs/ml/ml.duckdb"
artifact_dir = "outputs/ml/artifacts"
report_dir = "outputs/ml/reports"

[features]
windows = [5, 10, 20, 60, 120, 240]
feature_set_id = "vpa_d_sequence"
include_structure_state = false

[labels]
horizons = [1, 5, 10]
main_horizon = 5
risk_drawdown_threshold = -0.05

[split]
embargo_days = 10
folds = [
    { train_start = "2018-01-01", train_end = "2020-12-31", valid_start = "2021-01-01", valid_end = "2021-12-31", test_start = "2022-01-01", test_end = "2022-12-31" },
    { train_start = "2019-01-01", train_end = "2021-12-31", valid_start = "2022-01-01", valid_end = "2022-12-31", test_start = "2023-01-01", test_end = "2023-12-31" },
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
```

- [ ] Create `config/ml_vpa.toml` as the VPA generation profile for ML runs.

```toml
windows = [5, 10, 20, 60, 120, 240]

[trend_context.parent_windows]
5 = [20]
10 = [20, 60]
20 = [60]
60 = [240]
120 = [240]
```

- [ ] Implement `ml_stock_selector/config.py`.

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class MLConfig:
    data: dict[str, object]
    features: dict[str, object]
    labels: dict[str, object]
    split: dict[str, object]
    model: dict[str, object]
    portfolio: dict[str, object]


def load_ml_config(path: Path | str) -> MLConfig:
    raw = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    required = ["data", "features", "labels", "split", "model", "portfolio"]
    missing = [key for key in required if key not in raw]
    if missing:
        raise ValueError(f"Missing ML config sections: {', '.join(missing)}")
    return MLConfig(
        data=raw["data"],
        features=raw["features"],
        labels=raw["labels"],
        split=raw["split"],
        model=raw["model"],
        portfolio=raw["portfolio"],
    )
```

- [ ] Add CLI skeletons with `argparse` and `--help` support. Each CLI should accept `--config` and print parsed arguments before the corresponding implementation exists.

```python
from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/ml_default.toml")
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(args)


if __name__ == "__main__":
    main()
```

- [ ] Run baseline tests.

```bash
python -m pytest tests -v
python scripts/run_ml_feature_mart.py --help
python scripts/build_ml_labels.py --help
python scripts/train_ml_models.py --help
```

Expected: pytest passes; each help command exits with code 0.

- [ ] Commit.

```bash
git add pyproject.toml config/ml_default.toml config/ml_vpa.toml ml_stock_selector scripts tests
git commit -m "Introduce ML selector subsystem boundary" -m "The ML package starts as a sibling subsystem with explicit configuration and CLI entrypoints so VPA analytical rules remain untouched.\n\nConstraint: ML must consume vpa_* contracts instead of changing VPA rule modules\nConfidence: high\nScope-risk: narrow\nTested: python -m pytest tests -v; python scripts/run_ml_feature_mart.py --help; python scripts/build_ml_labels.py --help; python scripts/train_ml_models.py --help"
```

### Task 2: Add ML DuckDB Schema And Storage Helpers

**Files:**

- Create: `sql/create_ml_tables.sql`
- Create: `ml_stock_selector/storage.py`
- Create: `tests/test_ml_storage_schema.py`

- [ ] Create `sql/create_ml_tables.sql`.

```sql
create table if not exists ml_feature_mart_daily (
    trade_date varchar not null,
    code varchar not null,
    feature_set_id varchar not null,
    vpa_data_version varchar,
    generated_at varchar not null,
    industry_code varchar,
    industry_name varchar,
    is_st boolean,
    is_paused boolean,
    limit_up double,
    limit_down double,
    adv20_amount double,
    features_json varchar,
    primary key (trade_date, code, feature_set_id)
);

create table if not exists ml_labels_daily (
    trade_date varchar not null,
    code varchar not null,
    horizon_d integer not null,
    future_ret double,
    future_max_gain double,
    future_max_drawdown double,
    future_score double,
    future_rank_pct double,
    rank_label integer,
    risk_label integer,
    outperform_market boolean,
    generated_at varchar not null,
    primary key (trade_date, code, horizon_d)
);

create table if not exists ml_model_registry (
    model_id varchar not null,
    model_type varchar not null,
    feature_set_id varchar not null,
    label_name varchar not null,
    horizon_d integer not null,
    train_start varchar not null,
    train_end varchar not null,
    valid_start varchar not null,
    valid_end varchar not null,
    test_start varchar,
    test_end varchar,
    params_json varchar not null,
    metrics_json varchar,
    artifact_uri varchar not null,
    created_at varchar not null,
    primary key (model_id)
);

create table if not exists ml_predictions_daily (
    trade_date varchar not null,
    code varchar not null,
    model_id varchar not null,
    horizon_d integer not null,
    alpha_score double,
    alpha_rank_pct double,
    reg_score double,
    risk_score double,
    risk_rank_pct double,
    context_score double,
    liquidity_score double,
    penalty_score double,
    trade_score double,
    feature_set_id varchar not null,
    generated_at varchar not null,
    primary key (trade_date, code, model_id, horizon_d)
);

create table if not exists ml_portfolio_targets_daily (
    trade_date varchar not null,
    portfolio_id varchar not null,
    code varchar not null,
    target_weight double not null,
    rank_n integer not null,
    trade_score double not null,
    entry_reason varchar,
    generated_at varchar not null,
    primary key (trade_date, portfolio_id, code)
);

create table if not exists ml_backtest_orders (
    run_id varchar not null,
    sim_date varchar not null,
    decision_date varchar not null,
    code varchar not null,
    side varchar not null,
    target_weight double not null,
    order_px_ref varchar not null,
    fill_px double,
    status varchar not null,
    reason varchar,
    primary key (run_id, sim_date, code, side)
);

create table if not exists ml_backtest_positions (
    run_id varchar not null,
    sim_date varchar not null,
    code varchar not null,
    position_qty double,
    market_value double,
    weight double,
    primary key (run_id, sim_date, code)
);

create table if not exists ml_backtest_nav (
    run_id varchar not null,
    sim_date varchar not null,
    nav double not null,
    cash double not null,
    gross_exposure double not null,
    turnover double not null,
    primary key (run_id, sim_date)
);

create table if not exists ml_backtest_metrics (
    run_id varchar not null,
    metric_name varchar not null,
    metric_value double,
    segment varchar not null,
    primary key (run_id, metric_name, segment)
);
```

- [ ] Implement `ml_stock_selector/storage.py`.

```python
from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import duckdb
import pandas as pd


def init_ml_db(path: Path | str) -> duckdb.DuckDBPyConnection:
    db_path = Path(path)
    if str(path) != ":memory:":
        db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(path))
    schema_path = Path(__file__).resolve().parents[1] / "sql" / "create_ml_tables.sql"
    con.execute(schema_path.read_text(encoding="utf-8"))
    return con


def upsert_dataframe(
    con: duckdb.DuckDBPyConnection,
    table_name: str,
    frame: pd.DataFrame,
    key_columns: list[str],
) -> None:
    if frame.empty:
        return
    temp_name = f"_ml_upsert_{uuid4().hex}"
    con.register(temp_name, frame)
    condition = " and ".join(
        f"{table_name}.{column} = {temp_name}.{column}" for column in key_columns
    )
    try:
        con.execute(f"delete from {table_name} using {temp_name} where {condition}")
        con.execute(f"insert into {table_name} by name select * from {temp_name}")
    finally:
        con.unregister(temp_name)
```

- [ ] Add schema/upsert tests.

```python
from __future__ import annotations

import duckdb
import pandas as pd

from ml_stock_selector.storage import init_ml_db, upsert_dataframe


def test_init_ml_db_creates_project_owned_ml_tables(tmp_path):
    db_path = tmp_path / "ml.duckdb"
    con = init_ml_db(db_path)
    con.close()

    check = duckdb.connect(str(db_path))
    tables = {
        row[0]
        for row in check.execute(
            """
            select table_name
            from information_schema.tables
            where table_schema = 'main'
            """
        ).fetchall()
    }
    check.close()

    assert {
        "ml_feature_mart_daily",
        "ml_labels_daily",
        "ml_model_registry",
        "ml_predictions_daily",
        "ml_portfolio_targets_daily",
        "ml_backtest_orders",
        "ml_backtest_positions",
        "ml_backtest_nav",
        "ml_backtest_metrics",
    }.issubset(tables)


def test_ml_upsert_dataframe_replaces_existing_prediction_key(tmp_path):
    con = init_ml_db(tmp_path / "ml.duckdb")
    first = pd.DataFrame(
        [
            {
                "trade_date": "2024-01-02",
                "code": "000001.SZ",
                "model_id": "ranker-v1",
                "horizon_d": 5,
                "alpha_score": 0.2,
                "alpha_rank_pct": 0.8,
                "reg_score": None,
                "risk_score": 0.1,
                "risk_rank_pct": 0.2,
                "context_score": 70.0,
                "liquidity_score": 0.9,
                "penalty_score": 0.0,
                "trade_score": 0.7,
                "feature_set_id": "vpa_d_sequence",
                "generated_at": "2026-05-30T00:00:00",
            }
        ]
    )
    second = first.copy()
    second.loc[0, "trade_score"] = 0.9

    key = ["trade_date", "code", "model_id", "horizon_d"]
    upsert_dataframe(con, "ml_predictions_daily", first, key)
    upsert_dataframe(con, "ml_predictions_daily", second, key)

    row = con.execute("select count(*), max(trade_score) from ml_predictions_daily").fetchone()
    con.close()
    assert row == (1, 0.9)
```

- [ ] Run focused tests.

```bash
python -m pytest tests/test_ml_storage_schema.py -v
```

Expected: PASS.

- [ ] Commit.

```bash
git add sql/create_ml_tables.sql ml_stock_selector/storage.py tests/test_ml_storage_schema.py
git commit -m "Fix ML data contracts before model work" -m "The ML subsystem now owns explicit DuckDB tables and an idempotent storage helper aligned with existing VPA persistence patterns.\n\nConstraint: ML writes only ml_* tables\nConfidence: high\nScope-risk: narrow\nTested: python -m pytest tests/test_ml_storage_schema.py -v"
```

### Task 3: Build Feature Mart From Existing VPA Tables

**Files:**

- Create: `ml_stock_selector/feature_mart.py`
- Modify: `scripts/run_ml_feature_mart.py`
- Create: `tests/test_ml_feature_mart.py`

- [ ] Implement `build_feature_mart()` to read only stock-scope `vpa_*` rows and return one row per `trade_date, code`.

```python
from __future__ import annotations

from datetime import datetime, timezone
import json

import duckdb
import pandas as pd


def build_feature_mart(
    vpa_db_path: str,
    start_date: str,
    end_date: str,
    feature_set_id: str,
    include_structure_state: bool = False,
) -> pd.DataFrame:
    con = duckdb.connect(vpa_db_path, read_only=True)
    features = con.execute(
        """
        select *
        from vpa_features
        where scope_type = 'stock'
          and date between ? and ?
        order by date, scope_id, window_n
        """,
        [start_date, end_date],
    ).fetchdf()
    labels = con.execute(
        """
        select *
        from vpa_bar_context_labels
        where scope_type = 'stock'
          and date between ? and ?
        order by date, scope_id, window_n
        """,
        [start_date, end_date],
    ).fetchdf()
    sequences = con.execute(
        """
        select *
        from vpa_sequence_stats
        where scope_type = 'stock'
          and date between ? and ?
        order by date, scope_id, window_n
        """,
        [start_date, end_date],
    ).fetchdf()
    states = con.execute(
        """
        select *
        from vpa_structure_state
        where scope_type = 'stock'
          and date between ? and ?
        order by date, scope_id
        """,
        [start_date, end_date],
    ).fetchdf()
    con.close()

    if features.empty:
        return pd.DataFrame(
            columns=["trade_date", "code", "feature_set_id", "generated_at", "features_json"]
        )

    numeric = _pivot_by_window(
        features,
        ["ret_pct", "range_pct", "body_pct", "vol_rvol_n", "range_rvol_n", "price_position_n", "ma_slope_n"],
    )
    label_features = _pivot_by_window(
        labels,
        ["bull_bear_score", "supply_score", "demand_score", "volatility_score", "raw_label"],
    )
    sequence_features = _pivot_by_window(
        sequences,
        ["abnormal_ratio", "support_label_count", "supply_label_count", "bull_score_change", "sequence_strength_score", "sequence_pattern"],
    )

    wide = numeric.merge(label_features, on=["trade_date", "code"], how="left")
    wide = wide.merge(sequence_features, on=["trade_date", "code"], how="left")

    if include_structure_state and not states.empty:
        state_cols = [
            "date",
            "scope_id",
            "final_state",
            "trend_background",
            "position_background",
            "market_score",
            "sector_score",
            "self_score",
            "relative_strength_score",
            "resonance_score",
            "final_rating",
            "confidence",
        ]
        state_frame = states[state_cols].rename(columns={"date": "trade_date", "scope_id": "code"})
        wide = wide.merge(state_frame, on=["trade_date", "code"], how="left")

    generated_at = datetime.now(timezone.utc).isoformat()
    feature_cols = [col for col in wide.columns if col not in {"trade_date", "code"}]
    out = pd.DataFrame(
        {
            "trade_date": wide["trade_date"],
            "code": wide["code"],
            "feature_set_id": feature_set_id,
            "vpa_data_version": "v1",
            "generated_at": generated_at,
            "industry_code": None,
            "industry_name": None,
            "is_st": None,
            "is_paused": None,
            "limit_up": None,
            "limit_down": None,
            "adv20_amount": None,
            "features_json": wide[feature_cols].apply(
                lambda row: json.dumps(row.dropna().to_dict(), ensure_ascii=False, sort_keys=True),
                axis=1,
            ),
        }
    )
    return out.sort_values(["trade_date", "code"]).reset_index(drop=True)


def _pivot_by_window(frame: pd.DataFrame, value_columns: list[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["trade_date", "code"])
    rows = []
    for (date, code), group in frame.groupby(["date", "scope_id"], sort=False):
        row = {"trade_date": date, "code": code}
        for item in group.itertuples(index=False):
            window = int(item.window_n)
            for col in value_columns:
                if col in frame.columns:
                    row[f"{col}_{window}"] = getattr(item, col)
        rows.append(row)
    return pd.DataFrame(rows)
```

- [ ] Replace CLI skeleton in `scripts/run_ml_feature_mart.py` so it builds and writes `ml_feature_mart_daily` using `init_ml_db()` and `upsert_dataframe()`.

- [ ] Add tests that create a small VPA DuckDB with two stocks, two dates, and 5/20 windows; assert primary-key uniqueness and expected `features_json` keys such as `ret_pct_5`, `raw_label_20`, and `support_label_count_20`.

- [ ] Run focused tests and CLI help.

```bash
python -m pytest tests/test_ml_feature_mart.py -v
python scripts/run_ml_feature_mart.py --help
```

Expected: PASS; help exits 0.

- [ ] Commit.

```bash
git add ml_stock_selector/feature_mart.py scripts/run_ml_feature_mart.py tests/test_ml_feature_mart.py
git commit -m "Build ML feature mart from VPA contracts" -m "The feature mart pivots stock-scope vpa_* rows into one stock-date row while keeping VPA rule modules unchanged.\n\nConstraint: Feature mart reads vpa_* tables only\nConfidence: medium\nScope-risk: moderate\nTested: python -m pytest tests/test_ml_feature_mart.py -v; python scripts/run_ml_feature_mart.py --help"
```

### Task 4: Build Future Labels Without Decision-Time Leakage

**Files:**

- Create: `ml_stock_selector/label_builder.py`
- Modify: `scripts/build_ml_labels.py`
- Create: `tests/test_ml_label_builder.py`

- [ ] Implement labels for horizons `[1, 5, 10]`.

```python
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd


def build_labels(
    stock_prices: pd.DataFrame,
    horizons: list[int],
    risk_drawdown_threshold: float = -0.05,
) -> pd.DataFrame:
    required = {"date", "code", "close", "high", "low"}
    missing = sorted(required - set(stock_prices.columns))
    if missing:
        raise ValueError(f"Missing price columns: {', '.join(missing)}")

    rows: list[dict[str, object]] = []
    generated_at = datetime.now(timezone.utc).isoformat()
    for code, group in stock_prices.groupby("code", sort=False):
        prices = group.sort_values("date").reset_index(drop=True)
        for idx, item in prices.iterrows():
            current_close = float(item["close"])
            for horizon in horizons:
                future = prices.iloc[idx + 1 : idx + horizon + 1]
                if len(future) < horizon:
                    continue
                future_ret = _ret(current_close, float(future.iloc[-1]["close"]))
                future_max_gain = _ret(current_close, float(future["high"].max()))
                future_max_drawdown = _ret(current_close, float(future["low"].min()))
                future_score = future_ret + 0.5 * future_max_gain - 0.7 * abs(future_max_drawdown)
                rows.append(
                    {
                        "trade_date": item["date"],
                        "code": code,
                        "horizon_d": horizon,
                        "future_ret": future_ret,
                        "future_max_gain": future_max_gain,
                        "future_max_drawdown": future_max_drawdown,
                        "future_score": future_score,
                        "future_rank_pct": None,
                        "rank_label": None,
                        "risk_label": int(future_max_drawdown <= risk_drawdown_threshold),
                        "outperform_market": None,
                        "generated_at": generated_at,
                    }
                )

    labels = pd.DataFrame(rows)
    if labels.empty:
        return labels
    labels["future_rank_pct"] = labels.groupby(["trade_date", "horizon_d"])["future_score"].rank(pct=True)
    labels["rank_label"] = labels["future_rank_pct"].map(_rank_label).astype("int64")
    return labels.sort_values(["trade_date", "code", "horizon_d"]).reset_index(drop=True)


def _ret(current: float, future: float) -> float:
    return round(float(future) / float(current) - 1.0, 12)


def _rank_label(rank_pct: float) -> int:
    if rank_pct >= 0.8:
        return 4
    if rank_pct >= 0.6:
        return 3
    if rank_pct >= 0.4:
        return 2
    if rank_pct >= 0.2:
        return 1
    return 0
```

- [ ] Implement CLI to load stock bars through existing source adapter, build labels, and upsert `ml_labels_daily`.

- [ ] Add tests that prove labels use T+1..T+h only. Include a fixture where T-day high changes but future labels do not change.

- [ ] Run focused tests.

```bash
python -m pytest tests/test_ml_label_builder.py -v
python scripts/build_ml_labels.py --help
```

Expected: PASS; help exits 0.

- [ ] Commit.

```bash
git add ml_stock_selector/label_builder.py scripts/build_ml_labels.py tests/test_ml_label_builder.py
git commit -m "Create leakage-safe ML labels" -m "Future return, path, rank, and risk labels are built only from T+1 through horizon bars so they can supervise T-close decisions without same-day leakage.\n\nConstraint: Labels must not use same-day open execution assumptions\nConfidence: high\nScope-risk: moderate\nTested: python -m pytest tests/test_ml_label_builder.py -v; python scripts/build_ml_labels.py --help"
```

### Task 5: Add Walk-Forward Dataset Splits And LightGBM Groups

**Files:**

- Create: `ml_stock_selector/datasets.py`
- Create: `tests/test_ml_datasets.py`

- [ ] Implement `TrainValidTestSplit` and `build_lgbm_group()`.

```python
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class DateRange:
    start: str
    end: str


@dataclass(frozen=True)
class TrainValidTestSplit:
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame


def make_walk_forward_split(
    samples: pd.DataFrame,
    train_range: DateRange,
    valid_range: DateRange,
    test_range: DateRange,
    embargo_days: int,
) -> TrainValidTestSplit:
    frame = samples.copy()
    frame["trade_date_dt"] = pd.to_datetime(frame["trade_date"])
    train_end = pd.to_datetime(train_range.end)
    valid_start = pd.to_datetime(valid_range.start)
    test_start = pd.to_datetime(test_range.start)

    train = frame[
        (frame["trade_date"] >= train_range.start)
        & (frame["trade_date"] <= train_range.end)
        & (frame["trade_date_dt"] <= valid_start - pd.Timedelta(days=embargo_days))
    ]
    valid = frame[
        (frame["trade_date"] >= valid_range.start)
        & (frame["trade_date"] <= valid_range.end)
        & (frame["trade_date_dt"] <= test_start - pd.Timedelta(days=embargo_days))
    ]
    test = frame[(frame["trade_date"] >= test_range.start) & (frame["trade_date"] <= test_range.end)]
    return TrainValidTestSplit(
        train=train.drop(columns=["trade_date_dt"]).reset_index(drop=True),
        valid=valid.drop(columns=["trade_date_dt"]).reset_index(drop=True),
        test=test.drop(columns=["trade_date_dt"]).reset_index(drop=True),
    )


def build_lgbm_group(frame: pd.DataFrame) -> list[int]:
    counts = frame.sort_values(["trade_date", "code"]).groupby("trade_date", sort=True).size()
    return [int(value) for value in counts.tolist()]
```

- [ ] Add tests for embargo behavior and group sums.

```python
def test_build_lgbm_group_sums_to_sample_count():
    frame = pd.DataFrame(
        {
            "trade_date": ["2024-01-02", "2024-01-02", "2024-01-03"],
            "code": ["000001.SZ", "000002.SZ", "000001.SZ"],
        }
    )
    group = build_lgbm_group(frame)
    assert group == [2, 1]
    assert sum(group) == len(frame)
```

- [ ] Run focused tests.

```bash
python -m pytest tests/test_ml_datasets.py -v
```

Expected: PASS.

- [ ] Commit.

```bash
git add ml_stock_selector/datasets.py tests/test_ml_datasets.py
git commit -m "Make walk-forward dataset splits explicit" -m "Training, validation, and test slices now carry embargo handling and ranking group construction so LightGBM receives date-level query groups.\n\nConstraint: Adjacent horizon labels overlap and require embargo support\nConfidence: high\nScope-risk: narrow\nTested: python -m pytest tests/test_ml_datasets.py -v"
```

### Task 6: Train And Persist Minimum Alpha Ranker

**Files:**

- Create: `ml_stock_selector/models/artifacts.py`
- Create: `ml_stock_selector/models/alpha_ranker.py`
- Modify: `scripts/train_ml_models.py`
- Create: `tests/test_alpha_ranker.py`

- [ ] Implement a model artifact contract.

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelArtifact:
    model_id: str
    model_type: str
    feature_set_id: str
    label_name: str
    horizon_d: int
    feature_columns: list[str]
    artifact_dir: Path
    metrics: dict[str, float]
```

- [ ] Implement `train_alpha_ranker()` with `LGBMRanker`, date-sorted group arrays, feature importance export, and model reload support.

- [ ] Add a test using a synthetic panel with at least three dates and four stocks per date. Assert predictions length equals input rows and artifact reload produces predictions.

- [ ] Run focused tests.

```bash
python -m pytest tests/test_alpha_ranker.py -v
python scripts/train_ml_models.py --help
```

Expected: PASS; help exits 0.

- [ ] Commit.

```bash
git add ml_stock_selector/models/artifacts.py ml_stock_selector/models/alpha_ranker.py scripts/train_ml_models.py tests/test_alpha_ranker.py
git commit -m "Train the first cross-sectional alpha ranker" -m "The first ML model optimizes daily stock ranking with LightGBM group data and persists reloadable artifacts plus validation metrics.\n\nRejected: Binary up/down classifier as primary model | daily Top-10/15 selection is a ranking problem\nConfidence: medium\nScope-risk: moderate\nTested: python -m pytest tests/test_alpha_ranker.py -v; python scripts/train_ml_models.py --help"
```

### Task 7: Add Phase 1 Smoke Test

**Files:**

- Create: `tests/test_ml_pipeline_smoke.py`

- [ ] Add a synthetic end-to-end test that initializes VPA and ML temp databases, inserts minimal `vpa_*` rows, builds feature mart rows, builds labels from synthetic prices, joins samples, trains alpha ranker, and predicts.

- [ ] Run the Phase 1 verification set.

```bash
python -m pytest \
  tests/test_ml_storage_schema.py \
  tests/test_ml_feature_mart.py \
  tests/test_ml_label_builder.py \
  tests/test_ml_datasets.py \
  tests/test_alpha_ranker.py \
  tests/test_ml_pipeline_smoke.py \
  -v
```

Expected: PASS.

- [ ] Commit.

```bash
git add tests/test_ml_pipeline_smoke.py
git commit -m "Prove the ML data foundation works end to end" -m "A smoke test now covers ML schema initialization, feature mart construction, label building, sample assembly, ranker training, and prediction.\n\nConfidence: medium\nScope-risk: narrow\nTested: python -m pytest tests/test_ml_storage_schema.py tests/test_ml_feature_mart.py tests/test_ml_label_builder.py tests/test_ml_datasets.py tests/test_alpha_ranker.py tests/test_ml_pipeline_smoke.py -v"
```

---

## Phase 2: Decision-First Portfolio And Backtest

**Objective:** Convert model predictions into executable portfolio targets and simulate them with T+1 execution, transaction costs, tradeability filters, positions, NAV, and core metrics.

**Exit Criteria:**

- `python -m pytest tests/test_ml_scoring.py tests/test_portfolio_constructor.py tests/test_backtest_execution.py tests/test_backtest_engine.py tests/test_backtest_metrics.py -v` passes.
- `python scripts/run_ml_backtest.py --help` runs without crashing.
- A synthetic backtest proves that T-day predictions create T+1 fills, not same-day fills.

### Task 8: Add Prediction Scoring Formula

**Files:**

- Create: `ml_stock_selector/scoring.py`
- Create: `tests/test_ml_scoring.py`

- [ ] Implement cross-sectional percentile scoring and the recommended trade score formula.

```python
from __future__ import annotations

import pandas as pd


def score_candidates(predictions: pd.DataFrame) -> pd.DataFrame:
    required = {"trade_date", "code", "alpha_score", "risk_score", "context_score", "liquidity_score"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"Missing prediction columns: {', '.join(missing)}")

    out = predictions.copy()
    out["alpha_rank_pct"] = out.groupby("trade_date")["alpha_score"].rank(pct=True)
    out["risk_rank_pct"] = out.groupby("trade_date")["risk_score"].rank(pct=True)
    out["context_score_pct"] = out.groupby("trade_date")["context_score"].rank(pct=True)
    out["liquidity_score_pct"] = out.groupby("trade_date")["liquidity_score"].rank(pct=True)
    if "relative_strength_pct" not in out:
        out["relative_strength_pct"] = 0.5
    if "resonance_pct" not in out:
        out["resonance_pct"] = 0.5
    if "penalty_score" not in out:
        out["penalty_score"] = 0.0

    out["trade_score"] = (
        0.60 * out["alpha_rank_pct"]
        + 0.15 * out["context_score_pct"]
        + 0.10 * out["liquidity_score_pct"]
        + 0.05 * out["relative_strength_pct"]
        + 0.10 * out["resonance_pct"]
        - 0.30 * out["risk_rank_pct"]
        - out["penalty_score"]
    )
    return out.sort_values(["trade_date", "trade_score", "code"], ascending=[True, False, True]).reset_index(drop=True)
```

- [ ] Add tests for percentile ranking, risk penalty direction, and stable tie ordering by `code`.

- [ ] Run focused tests.

```bash
python -m pytest tests/test_ml_scoring.py -v
```

Expected: PASS.

- [ ] Commit.

```bash
git add ml_stock_selector/scoring.py tests/test_ml_scoring.py
git commit -m "Turn model outputs into trade scores" -m "Candidate scoring now combines alpha, context, liquidity, relative strength, resonance, and risk penalties with daily percentile normalization.\n\nConstraint: ML alpha dominates but VPA context remains a controlled adjustment\nConfidence: medium\nScope-risk: narrow\nTested: python -m pytest tests/test_ml_scoring.py -v"
```

### Task 9: Construct Constrained Portfolio Targets

**Files:**

- Create: `ml_stock_selector/portfolio/constraints.py`
- Create: `ml_stock_selector/portfolio/constructor.py`
- Create: `ml_stock_selector/portfolio/allocator.py`
- Create: `tests/test_portfolio_constructor.py`

- [ ] Implement hard filters for ST, paused, insufficient liquidity, and unavailable next-open buys.

- [ ] Implement target selection with defaults: `target_positions=12`, `hard_max_positions=15`, `max_industry_names=3`, `max_new_entries_per_day=4`, `allow_cash=true`.

- [ ] Implement bounded equal-weight allocation capped by `single_name_min_weight` and `single_name_max_weight`.

- [ ] Add tests proving:

```text
1. ST and paused names are excluded.
2. A date with too few passing candidates returns fewer than target_positions.
3. No industry exceeds max_industry_names.
4. No output exceeds hard_max_positions.
5. Weights are between 0.05 and 0.10 when enough names exist.
```

- [ ] Run focused tests.

```bash
python -m pytest tests/test_portfolio_constructor.py -v
```

Expected: PASS.

- [ ] Commit.

```bash
git add ml_stock_selector/portfolio tests/test_portfolio_constructor.py
git commit -m "Build constrained portfolio targets from trade scores" -m "Portfolio construction now converts ranked candidates into realistic 10-15 name targets while respecting tradeability, industry concentration, and cash allowance.\n\nConstraint: Weak opportunity sets must allow cash instead of forced buying\nConfidence: medium\nScope-risk: moderate\nTested: python -m pytest tests/test_portfolio_constructor.py -v"
```

### Task 10: Simulate T+1 Execution

**Files:**

- Create: `ml_stock_selector/backtest/execution.py`
- Create: `tests/test_backtest_execution.py`

- [ ] Implement next-open execution where `decision_date` maps to the next available trading date per stock.

- [ ] Apply buy/sell side, slippage, commission, stamp duty, paused filter, and limit-up buy rejection.

- [ ] Add tests proving:

```text
1. A target generated on 2024-01-02 cannot fill on 2024-01-02.
2. The earliest valid fill date is 2024-01-03 when a 2024-01-03 bar exists.
3. A paused stock produces status="rejected" with reason="paused".
4. A buy at next open equal to limit_up produces status="rejected" with reason="limit_up".
5. Filled price includes configured slippage.
```

- [ ] Run focused tests.

```bash
python -m pytest tests/test_backtest_execution.py -v
```

Expected: PASS.

- [ ] Commit.

```bash
git add ml_stock_selector/backtest/execution.py tests/test_backtest_execution.py
git commit -m "Enforce T+1 execution in ML backtests" -m "Execution simulation now rejects same-day fills and applies tradeability, limit, and cost rules at the next tradable open.\n\nConstraint: T-close decisions must not be backfilled into same-day open execution\nConfidence: high\nScope-risk: moderate\nTested: python -m pytest tests/test_backtest_execution.py -v"
```

### Task 11: Add Backtest Engine And NAV Accounting

**Files:**

- Create: `ml_stock_selector/backtest/engine.py`
- Modify: `scripts/run_ml_backtest.py`
- Create: `tests/test_backtest_engine.py`

- [ ] Implement a small `BacktestConfig` dataclass with `initial_cash`, `execution_price`, `slippage_bps`, `commission_bps`, `stamp_duty_bps`, and `portfolio_id`.

- [ ] Implement `run_backtest()` to process targets by decision date, call execution, update holdings and cash, mark positions to close, and output orders, positions, and NAV frames.

- [ ] Add tests proving NAV decreases by costs after fills and cash remains when fewer candidates pass.

- [ ] Update `scripts/run_ml_backtest.py` with `--config`, `--predictions-db`, `--bars-source`, `--start-date`, `--end-date`, `--output-db`, and `--run-id`.

- [ ] Run focused tests and CLI help.

```bash
python -m pytest tests/test_backtest_engine.py -v
python scripts/run_ml_backtest.py --help
```

Expected: PASS; help exits 0.

- [ ] Commit.

```bash
git add ml_stock_selector/backtest/engine.py scripts/run_ml_backtest.py tests/test_backtest_engine.py
git commit -m "Close the prediction-to-NAV backtest loop" -m "The backtest engine now turns daily targets into orders, holdings, cash, and NAV while preserving T+1 execution semantics.\n\nConfidence: medium\nScope-risk: moderate\nTested: python -m pytest tests/test_backtest_engine.py -v; python scripts/run_ml_backtest.py --help"
```

### Task 12: Add Backtest Metrics

**Files:**

- Create: `ml_stock_selector/backtest/metrics.py`
- Create: `ml_stock_selector/backtest/reports.py`
- Create: `tests/test_backtest_metrics.py`

- [ ] Implement metric functions for `RankIC`, `NDCG@10`, `NDCG@15`, Top-N mean return, annualized return, max drawdown, turnover, and profit/loss ratio.

- [ ] Add a report writer that emits CSV metrics and yearly slices under `outputs/ml/reports`.

- [ ] Add tests with hand-calculated NAV and prediction fixtures.

- [ ] Run Phase 2 verification.

```bash
python -m pytest \
  tests/test_ml_scoring.py \
  tests/test_portfolio_constructor.py \
  tests/test_backtest_execution.py \
  tests/test_backtest_engine.py \
  tests/test_backtest_metrics.py \
  -v
```

Expected: PASS.

- [ ] Commit.

```bash
git add ml_stock_selector/backtest/metrics.py ml_stock_selector/backtest/reports.py tests/test_backtest_metrics.py
git commit -m "Report ranking and portfolio backtest quality" -m "Backtests now expose ranking, Top-N, NAV, drawdown, turnover, and yearly-slice metrics needed to judge ML selector quality.\n\nConfidence: medium\nScope-risk: narrow\nTested: python -m pytest tests/test_ml_scoring.py tests/test_portfolio_constructor.py tests/test_backtest_execution.py tests/test_backtest_engine.py tests/test_backtest_metrics.py -v"
```

---

## Phase 3: Governance, Experiments, Reports

**Objective:** Make model outputs reproducible and comparable across feature ablations, model types, folds, and reports.

**Exit Criteria:**

- `python -m pytest tests/test_alpha_regressor.py tests/test_risk_model.py tests/test_ml_pipeline_smoke.py -v` passes.
- Every trained model writes a registry row with feature set, label, horizon, parameters, metrics, and artifact URI.
- A/B/C/D/E feature ablation runs can be represented by config and report output.

### Task 13: Add Regressor And Risk Model

**Files:**

- Create: `ml_stock_selector/models/alpha_regressor.py`
- Create: `ml_stock_selector/models/risk_model.py`
- Create: `tests/test_alpha_regressor.py`
- Create: `tests/test_risk_model.py`

- [ ] Implement `train_alpha_regressor()` using LightGBM regression on `future_score_5d`.

- [ ] Implement `train_risk_model()` using LightGBM binary classification on `risk_label_5d`.

- [ ] Reuse `ModelArtifact` and feature-column persistence from Task 6.

- [ ] Add synthetic tests proving both models train, predict, save, reload, and return deterministic prediction lengths.

- [ ] Run focused tests.

```bash
python -m pytest tests/test_alpha_regressor.py tests/test_risk_model.py -v
```

Expected: PASS.

- [ ] Commit.

```bash
git add ml_stock_selector/models/alpha_regressor.py ml_stock_selector/models/risk_model.py tests/test_alpha_regressor.py tests/test_risk_model.py
git commit -m "Add auxiliary return and risk models" -m "Regression and risk classifiers complement the ranker while sharing the same artifact contract for reproducible scoring.\n\nRejected: Replacing the ranker with regression | Top-K portfolio selection still requires cross-sectional ranking as the primary objective\nConfidence: medium\nScope-risk: moderate\nTested: python -m pytest tests/test_alpha_regressor.py tests/test_risk_model.py -v"
```

### Task 14: Add Registry And Calibration

**Files:**

- Create: `ml_stock_selector/registry.py`
- Create: `ml_stock_selector/models/calibrator.py`
- Modify: `scripts/train_ml_models.py`
- Create: `tests/test_ml_registry.py`

- [ ] Implement `register_model()` to upsert `ml_model_registry`.

- [ ] Implement cross-sectional percentile calibration for model outputs by date.

- [ ] Ensure every training CLI run writes `params_json`, `metrics_json`, `artifact_uri`, `feature_set_id`, and date ranges.

- [ ] Add tests proving `model_id` uniqueness and registry upsert behavior.

- [ ] Run focused tests.

```bash
python -m pytest tests/test_ml_registry.py -v
```

Expected: PASS.

- [ ] Commit.

```bash
git add ml_stock_selector/registry.py ml_stock_selector/models/calibrator.py scripts/train_ml_models.py tests/test_ml_registry.py
git commit -m "Register and calibrate model artifacts" -m "Model outputs are now traceable to parameters, feature sets, labels, date ranges, metrics, and artifact paths.\n\nConstraint: Daily predictions must be reproducible from registry metadata\nConfidence: medium\nScope-risk: moderate\nTested: python -m pytest tests/test_ml_registry.py -v"
```

### Task 15: Encode Feature Ablation Matrix

**Files:**

- Modify: `config/ml_default.toml`
- Modify: `ml_stock_selector/feature_mart.py`
- Create: `tests/test_ml_feature_sets.py`

- [ ] Add explicit feature set definitions for:

```text
baseline_a_ohlcv
baseline_b_vpa_numeric
vpa_c_bar_context
vpa_d_sequence
vpa_e_structure_state
```

- [ ] Implement feature set filtering so VPA E is the only set that includes `final_state`, `final_rating`, and other high-level structure-state fields.

- [ ] Add tests proving VPA C includes objective bar context labels, VPA D includes sequence stats, and VPA E includes structure state fields.

- [ ] Run focused tests.

```bash
python -m pytest tests/test_ml_feature_sets.py tests/test_ml_feature_mart.py -v
```

Expected: PASS.

- [ ] Commit.

```bash
git add config/ml_default.toml ml_stock_selector/feature_mart.py tests/test_ml_feature_sets.py tests/test_ml_feature_mart.py
git commit -m "Make VPA feature ablations executable" -m "Feature sets now distinguish numeric, objective label, sequence, and high-level structure layers so sample-out gain can be measured instead of assumed.\n\nDirective: Do not make final_state or final_rating default main features without ablation evidence\nConfidence: high\nScope-risk: moderate\nTested: python -m pytest tests/test_ml_feature_sets.py tests/test_ml_feature_mart.py -v"
```

### Task 16: Document And Smoke-Test The Full ML Loop

**Files:**

- Modify: `README.md`
- Modify: `tests/test_ml_pipeline_smoke.py`

- [ ] Add a README section named `ML Stock Selector Subsystem` with:

```text
1. Scope boundary: ML reads vpa_* and writes ml_*.
2. Required command order: VPA pipeline, feature mart, labels, train, backtest, daily signal.
3. T+1 execution rule.
4. Feature ablation matrix.
5. Generated output locations under outputs/ml/.
```

- [ ] Extend smoke test to run feature mart, labels, ranker, scoring, portfolio construction, execution, and metrics on fixtures.

- [ ] Run full tests.

```bash
python -m pytest tests -v
```

Expected: PASS.

- [ ] Commit.

```bash
git add README.md tests/test_ml_pipeline_smoke.py
git commit -m "Document and smoke-test the ML selector loop" -m "The repo now explains the ML subsystem boundary and verifies a minimal data-to-backtest path in tests.\n\nConfidence: medium\nScope-risk: narrow\nTested: python -m pytest tests -v"
```

---

## Phase 4: Daily Signal And Split Readiness

**Objective:** Turn validated model artifacts into daily signal outputs and leave the system ready for eventual independent repo extraction when contracts stabilize.

**Exit Criteria:**

- `python -m pytest tests/test_daily_signal.py tests/test_ml_pipeline_smoke.py -v` passes.
- `python scripts/run_ml_daily_signal.py --help` runs without crashing.
- Daily inference can load an active artifact, score the latest feature mart date, write `ml_predictions_daily`, and write `ml_portfolio_targets_daily`.
- Repo split readiness is documented as a checklist with concrete gates.

### Task 17: Load Active Artifacts For Inference

**Files:**

- Create: `ml_stock_selector/serving/artifact_loader.py`
- Create: `tests/test_daily_signal.py`

- [ ] Implement `load_active_model()` that reads `ml_model_registry`, selects a model by `model_type`, `feature_set_id`, `horizon_d`, and active model id, then loads the artifact path.

- [ ] Raise a clear `ValueError` when the registry has no matching active model.

- [ ] Add tests for successful load and missing model error text.

- [ ] Run focused tests.

```bash
python -m pytest tests/test_daily_signal.py -v
```

Expected: PASS.

- [ ] Commit.

```bash
git add ml_stock_selector/serving/artifact_loader.py tests/test_daily_signal.py
git commit -m "Load registered model artifacts for daily inference" -m "Serving code now resolves active model artifacts through ml_model_registry instead of hardcoded paths.\n\nConstraint: Daily predictions must be traceable to registry metadata\nConfidence: medium\nScope-risk: narrow\nTested: python -m pytest tests/test_daily_signal.py -v"
```

### Task 18: Generate Daily Signals

**Files:**

- Create: `ml_stock_selector/serving/daily_signal.py`
- Modify: `scripts/run_ml_daily_signal.py`
- Modify: `tests/test_daily_signal.py`

- [ ] Implement `generate_daily_signal()` to:

```text
1. Load latest feature mart rows for as_of_date.
2. Load active ranker, optional regressor, and optional risk model.
3. Produce raw scores and calibrated percentiles.
4. Apply score_candidates().
5. Apply portfolio constructor.
6. Upsert ml_predictions_daily and ml_portfolio_targets_daily.
```

- [ ] Implement CLI flags: `--config`, `--as-of-date`, `--ml-db`, `--model-id`, and `--portfolio-id`.

- [ ] Add tests proving no training is invoked during daily signal generation.

- [ ] Run focused tests and CLI help.

```bash
python -m pytest tests/test_daily_signal.py -v
python scripts/run_ml_daily_signal.py --help
```

Expected: PASS; help exits 0.

- [ ] Commit.

```bash
git add ml_stock_selector/serving/daily_signal.py scripts/run_ml_daily_signal.py tests/test_daily_signal.py
git commit -m "Generate daily ML portfolio signals" -m "Daily inference now loads registered artifacts, scores the latest feature mart rows, and writes both predictions and portfolio targets.\n\nConfidence: medium\nScope-risk: moderate\nTested: python -m pytest tests/test_daily_signal.py -v; python scripts/run_ml_daily_signal.py --help"
```

### Task 19: Add Operational Guardrails And Contract Checks

**Files:**

- Create: `tests/test_ml_contracts.py`
- Modify: `ml_stock_selector/feature_mart.py`
- Modify: `ml_stock_selector/serving/daily_signal.py`
- Modify: `ml_stock_selector/backtest/execution.py`

- [ ] Add explicit errors for missing required upstream `vpa_*` tables or missing feature columns.

- [ ] Add a contract test that fails if `ml_stock_selector` is imported by `vpa_structure_recognizer`.

- [ ] Add a leakage guard test that scans execution outputs and asserts `sim_date > decision_date` for all filled orders.

- [ ] Run contract tests.

```bash
python -m pytest tests/test_ml_contracts.py tests/test_backtest_execution.py -v
```

Expected: PASS.

- [ ] Commit.

```bash
git add tests/test_ml_contracts.py ml_stock_selector/feature_mart.py ml_stock_selector/serving/daily_signal.py ml_stock_selector/backtest/execution.py
git commit -m "Add ML subsystem contract guardrails" -m "Contract tests now protect the VPA-to-ML dependency direction and the T+1 execution rule.\n\nDirective: Never let VPA depend on ml_stock_selector modules\nConfidence: high\nScope-risk: narrow\nTested: python -m pytest tests/test_ml_contracts.py tests/test_backtest_execution.py -v"
```

### Task 20: Write Split-Readiness Checklist

**Files:**

- Create: `docs/superpowers/specs/ml-stock-selector-split-readiness.md`
- Modify: `README.md`

- [ ] Create a checklist with these concrete gates:

```text
1. vpa_* schema has no breaking change for six weeks.
2. ml_feature_mart_daily and ml_predictions_daily contracts are stable across at least two walk-forward runs.
3. Daily signal CLI reads only vpa_* or Parquet snapshots plus ml_model_registry artifacts.
4. No ml_stock_selector code imports private VPA implementation modules for training or serving.
5. Artifact directory structure and model_id naming are stable.
6. Backtest and daily inference both use the same prediction and portfolio target contracts.
7. Release cadence or team ownership requires independent packaging.
```

- [ ] Link the checklist from the README ML section.

- [ ] Run final verification.

```bash
python -m pytest tests -v
```

Expected: PASS.

- [ ] Commit.

```bash
git add docs/superpowers/specs/ml-stock-selector-split-readiness.md README.md
git commit -m "Define ML selector split readiness gates" -m "Repo extraction is now governed by stable contract and operational gates instead of an early architectural split.\n\nConfidence: high\nScope-risk: narrow\nTested: python -m pytest tests -v"
```

---

## Phase-Level Verification Commands

Run these after each phase.

```bash
# Phase 1
python -m pytest \
  tests/test_ml_storage_schema.py \
  tests/test_ml_feature_mart.py \
  tests/test_ml_label_builder.py \
  tests/test_ml_datasets.py \
  tests/test_alpha_ranker.py \
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
  tests/test_ml_pipeline_smoke.py \
  -v

# Phase 4 and final
python -m pytest \
  tests/test_daily_signal.py \
  tests/test_ml_contracts.py \
  tests/test_ml_pipeline_smoke.py \
  -v

python -m pytest tests -v
```

## Remaining Risks To Track During Implementation

- LightGBM dependency may need platform-specific installation handling in CI.
- Synthetic tests prove contracts, but real-market performance requires external data coverage and walk-forward runs.
- Feature mart initially serializes wide features into `features_json` for schema stability; if performance becomes a bottleneck, promote stable high-value columns to physical DuckDB columns with a migration.
- `config/ml_vpa.toml` adds a 5-day VPA window; implementation must verify existing VPA config parser accepts integer-like TOML keys in the parent window section.
- The first backtest engine is intentionally lightweight. Broker-specific fill rules, intraday VWAP, and order book effects are outside this four-phase plan.

