# Local VPA UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local Streamlit workbench for running, browsing, filtering, and explaining VPA results while keeping all analytical judgments deterministic.

**Architecture:** Keep the existing CLI and pipeline intact, add a service layer as the stable boundary for UI calls, then build a Streamlit app on top. Market scope selection is implemented as an explicit pipeline/service parameter and stored with each run; LLM remains outside the core judgment path.

**Tech Stack:** Python 3.11+, DuckDB Python API, pandas, openpyxl, Streamlit, pytest. No Tushare download logic belongs in the UI path; upstream DuckDB tables are read-only inputs.

---

## Scope Boundary

First-version scope:

- Run the existing VPA pipeline from a local UI.
- Read results from generated `vpa_*` DuckDB tables.
- Add explicit market-scope selection:
  - self-calculated all-A aggregate
  - selected upstream index
  - mixed mode: selected index trend plus self-calculated breadth
- Record market-scope metadata for each run.
- Show current sector source as local industry aggregation.
- Expose planned-but-unavailable index candidates such as `700081.TI` without allowing execution until local upstream bars exist.
- Generate deterministic Chinese stock explanations from existing fields.

Out of scope for first version:

- Public deployment, auth, user accounts, permissions.
- Automatic trading or trading advice.
- Tushare data download inside this repo.
- Writing derived results back to upstream DuckDB files.
- LLM-based scoring, labels, screening, or signal generation.
- Full strategy backtesting platform.

Backlog / upstream prerequisites:

- Sync Tushare `ths_index`/`ths_daily` for `700081.TI` or other THS market indexes into the upstream research DuckDB.
- Sync Tushare `index_basic(market='SW')`, `index_classify(src='SW2021')`, `index_member_all(...)`, and `sw_daily(...)` into an upstream sector-index source table.
- Add pipeline support for formal SW industry index bars after upstream tables exist.

## Target File Structure

- Modify: `pyproject.toml`
- Modify: `sql/create_vpa_tables.sql`
- Modify: `vpa_structure_recognizer/models.py`
- Modify: `vpa_structure_recognizer/data_sources.py`
- Modify: `vpa_structure_recognizer/market_aggregates.py`
- Modify: `vpa_structure_recognizer/pipeline.py`
- Modify: `vpa_structure_recognizer/report_formatter.py`
- Modify: `scripts/run_vpa_structure.py`
- Create: `vpa_structure_recognizer/app_service.py`
- Create: `apps/local_vpa_ui.py`
- Create: `tests/test_app_service.py`
- Modify: `tests/test_data_sources.py`
- Modify: `tests/test_market_aggregates.py`
- Modify: `tests/test_pipeline_smoke.py`
- Modify: `tests/test_report_formatter.py`
- Modify: `README.md`

## Shared Contracts

Use these names consistently across tasks:

```python
MARKET_MODE_SELF = "self_calculated_all_a"
MARKET_MODE_INDEX = "selected_index"
MARKET_MODE_MIXED = "mixed_index_breadth"

SECTOR_MODE_LOCAL = "local_industry_aggregate"

PLANNED_MARKET_INDEXES = {
    "700081.TI": {
        "name": "用户指定同花顺全市场候选",
        "source": "ths_daily",
        "status": "upstream_required",
    }
}
```

`market_mode` must be one of `self_calculated_all_a`, `selected_index`, or `mixed_index_breadth`.

`market_index_code` is required for `selected_index` and `mixed_index_breadth`.

`sector_mode` is `local_industry_aggregate` in the first version.

## Task 1: Add Streamlit Dependency and Preserve CLI Baseline

**Files:**
- Modify: `pyproject.toml`
- Test: `python -m pytest tests -v`

- [ ] Add project dependencies explicitly to `pyproject.toml`.

```toml
[project]
name = "vpa-structure-recognizer"
version = "0.1.0"
description = "A-share multi-level volume-price structure recognizer"
requires-python = ">=3.11"
dependencies = [
    "duckdb",
    "openpyxl",
    "pandas",
    "streamlit",
]

[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests"]
```

- [ ] Run full tests to prove dependency metadata did not change behavior.

Run:

```bash
python -m pytest tests -v
```

Expected: all existing tests pass.

- [ ] Commit.

```bash
git add pyproject.toml
git commit -m "Declare local UI runtime dependencies" -m "Streamlit is required for the local workbench while existing pipeline behavior remains unchanged.\n\nConfidence: high\nScope-risk: narrow\nTested: python -m pytest tests -v"
```

## Task 2: Add Run Metadata Storage

**Files:**
- Modify: `sql/create_vpa_tables.sql`
- Modify: `vpa_structure_recognizer/storage.py`
- Test: `tests/test_storage_schema.py`

- [ ] Add a `vpa_run_metadata` table to `sql/create_vpa_tables.sql`.

```sql
create table if not exists vpa_run_metadata (
    run_id varchar not null,
    created_at varchar not null,
    start_date varchar not null,
    end_date varchar not null,
    as_of_date varchar,
    source varchar,
    output_db varchar not null,
    report_path varchar,
    market_mode varchar not null,
    market_index_code varchar,
    market_scope_id varchar not null,
    sector_mode varchar not null,
    primary key (run_id)
);
```

- [ ] Add a storage helper in `vpa_structure_recognizer/storage.py`.

```python
def insert_run_metadata(con: duckdb.DuckDBPyConnection, metadata: dict[str, object]) -> None:
    frame = pd.DataFrame([metadata])
    upsert_dataframe(con, "vpa_run_metadata", frame, ["run_id"])
```

- [ ] Extend `tests/test_storage_schema.py` to verify the table exists and can upsert one metadata row.

```python
def test_run_metadata_table_accepts_upsert(tmp_path):
    con = init_vpa_db(tmp_path / "vpa.duckdb")
    insert_run_metadata(
        con,
        {
            "run_id": "run-1",
            "created_at": "2026-05-28T00:00:00",
            "start_date": "2024-01-01",
            "end_date": "2024-01-31",
            "as_of_date": "2024-01-31",
            "source": "research_source",
            "output_db": str(tmp_path / "vpa.duckdb"),
            "report_path": str(tmp_path / "report.xlsx"),
            "market_mode": "self_calculated_all_a",
            "market_index_code": None,
            "market_scope_id": "ALL_A",
            "sector_mode": "local_industry_aggregate",
        },
    )
    row = con.execute("select market_mode, sector_mode from vpa_run_metadata").fetchone()
    assert row == ("self_calculated_all_a", "local_industry_aggregate")
    con.close()
```

- [ ] Run focused tests.

```bash
python -m pytest tests/test_storage_schema.py -v
```

Expected: PASS.

- [ ] Commit.

```bash
git add sql/create_vpa_tables.sql vpa_structure_recognizer/storage.py tests/test_storage_schema.py
git commit -m "Record VPA run data-scope metadata" -m "Each run now stores the market and sector data scope so result files cannot be compared without knowing their judgment basis.\n\nConfidence: high\nScope-risk: narrow\nTested: python -m pytest tests/test_storage_schema.py -v"
```

## Task 3: Add Market Index Discovery to Data Sources

**Files:**
- Modify: `vpa_structure_recognizer/models.py`
- Modify: `vpa_structure_recognizer/data_sources.py`
- Modify: `tests/test_data_sources.py`

- [ ] Add index candidate columns in `vpa_structure_recognizer/models.py`.

```python
INDEX_CANDIDATE_COLUMNS = [
    "index_code",
    "name",
    "market",
    "publisher",
    "category",
    "min_date",
    "max_date",
    "row_count",
    "status",
    "source",
]

INDEX_BAR_COLUMNS = [
    "date",
    "index_code",
    "open",
    "high",
    "low",
    "close",
    "prev_close",
    "volume",
    "amount",
]
```

- [ ] Add `list_market_indexes()` and `fetch_index_bars()` to `ResearchSourceDuckDB`.

```python
def list_market_indexes(self) -> pd.DataFrame:
    con = duckdb.connect(str(self.path), read_only=True)
    frame = con.execute(
        """
        with coverage as (
            select
                index_code,
                min(trade_date) as min_date,
                max(trade_date) as max_date,
                count(*) as row_count
            from index_daily_bar_pit
            group by index_code
        )
        select
            coalesce(b.ts_code, c.index_code) as index_code,
            b.name,
            b.market,
            b.publisher,
            b.category,
            c.min_date,
            c.max_date,
            coalesce(c.row_count, 0) as row_count,
            case when c.row_count > 0 then 'available' else 'no_bars' end as status,
            'index_daily_bar_pit' as source
        from index_basic_ref b
        full outer join coverage c on c.index_code = b.ts_code
        order by index_code
        """
    ).fetchdf()
    con.close()
    planned = pd.DataFrame(
        [
            {
                "index_code": "700081.TI",
                "name": "用户指定同花顺全市场候选",
                "market": "THS",
                "publisher": "同花顺",
                "category": "宽基指数",
                "min_date": None,
                "max_date": None,
                "row_count": 0,
                "status": "upstream_required",
                "source": "ths_daily",
            }
        ]
    )
    return _nullable_frame(pd.concat([frame, planned], ignore_index=True)[INDEX_CANDIDATE_COLUMNS])
```

```python
def fetch_index_bars(self, index_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    start = _compact_date(start_date)
    end = _compact_date(end_date)
    con = duckdb.connect(str(self.path), read_only=True)
    frame = con.execute(
        """
        select
            substr(trade_date, 1, 4) || '-' || substr(trade_date, 5, 2) || '-' || substr(trade_date, 7, 2) as date,
            index_code,
            open,
            high,
            low,
            close,
            pre_close as prev_close,
            volume,
            turnover_value as amount
        from index_daily_bar_pit
        where index_code = ?
          and trade_date between ? and ?
        order by trade_date
        """,
        [index_code, start, end],
    ).fetchdf()
    con.close()
    return _nullable_frame(frame[INDEX_BAR_COLUMNS])
```

- [ ] Add explicit index behavior to `AuditedStockDuckDB`.

```python
def list_market_indexes(self) -> pd.DataFrame:
    return _nullable_frame(
        pd.DataFrame(
            [
                {
                    "index_code": "700081.TI",
                    "name": "用户指定同花顺全市场候选",
                    "market": "THS",
                    "publisher": "同花顺",
                    "category": "宽基指数",
                    "min_date": None,
                    "max_date": None,
                    "row_count": 0,
                    "status": "upstream_required",
                    "source": "ths_daily",
                }
            ],
            columns=INDEX_CANDIDATE_COLUMNS,
        )
    )


def fetch_index_bars(self, index_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    return _nullable_frame(pd.DataFrame(columns=INDEX_BAR_COLUMNS))
```

- [ ] Add test fixture tables for `index_basic_ref` and `index_daily_bar_pit`.

```python
def test_research_source_lists_available_and_planned_market_indexes(tmp_path):
    db_path = tmp_path / "source.duckdb"
    con = duckdb.connect(str(db_path))
    con.execute(
        "create table index_basic_ref (ts_code varchar, name varchar, market varchar, publisher varchar, category varchar, base_date varchar, base_point double, list_date varchar)"
    )
    con.execute(
        "create table index_daily_bar_pit (index_code varchar, trade_date varchar, open double, high double, low double, close double, pre_close double, change double, pct_chg double, volume double, turnover_value double, source_table varchar, ingested_at timestamp)"
    )
    con.execute("insert into index_basic_ref values ('000985.CSI', '中证全指', 'CSI', '中证指数有限公司', '规模指数', '20041231', 1000, '20110802')")
    con.execute("insert into index_basic_ref values ('700001.TI', '同花顺全A(加权)', 'THS', '同花顺', '宽基指数', null, null, null)")
    con.execute("insert into index_daily_bar_pit values ('000985.CSI', '20240102', 1, 2, 0.5, 1.5, 1.0, 0.5, 50, 100, 200, 'fixture', null)")
    con.close()

    indexes = ResearchSourceDuckDB(db_path).list_market_indexes()

    available = indexes[indexes["index_code"] == "000985.CSI"].iloc[0]
    no_bars = indexes[indexes["index_code"] == "700001.TI"].iloc[0]
    planned = indexes[indexes["index_code"] == "700081.TI"].iloc[0]
    assert available["status"] == "available"
    assert available["row_count"] == 1
    assert no_bars["status"] == "no_bars"
    assert planned["status"] == "upstream_required"
```

- [ ] Add fetch test.

```python
def test_research_source_fetches_index_bars(tmp_path):
    db_path = tmp_path / "source.duckdb"
    con = duckdb.connect(str(db_path))
    con.execute(
        "create table index_daily_bar_pit (index_code varchar, trade_date varchar, open double, high double, low double, close double, pre_close double, change double, pct_chg double, volume double, turnover_value double, source_table varchar, ingested_at timestamp)"
    )
    con.execute("insert into index_daily_bar_pit values ('000985.CSI', '20240102', 1, 2, 0.5, 1.5, 1.0, 0.5, 50, 100, 200, 'fixture', null)")
    con.close()

    bars = ResearchSourceDuckDB(db_path).fetch_index_bars("000985.CSI", "2024-01-01", "2024-01-31")

    assert bars.iloc[0]["date"] == "2024-01-02"
    assert bars.iloc[0]["amount"] == 200
```

- [ ] Run focused tests.

```bash
python -m pytest tests/test_data_sources.py -v
```

Expected: PASS.

- [ ] Commit.

```bash
git add vpa_structure_recognizer/models.py vpa_structure_recognizer/data_sources.py tests/test_data_sources.py
git commit -m "Discover executable and planned market indexes" -m "The app service can distinguish indexes with local daily bars from requested candidates that need upstream ingestion.\n\nConstraint: 700081.TI is planned but unavailable in the current research source\nConfidence: high\nScope-risk: moderate\nTested: python -m pytest tests/test_data_sources.py -v"
```

## Task 4: Add Market Bar Selection Helpers

**Files:**
- Modify: `vpa_structure_recognizer/market_aggregates.py`
- Modify: `tests/test_market_aggregates.py`

- [ ] Add a helper that converts index bars to the market aggregate contract.

```python
def build_market_bars_from_index(index_bars: pd.DataFrame, scope_id: str) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "date": index_bars["date"],
            "all_a_equal_weight_open": index_bars["open"],
            "all_a_equal_weight_high": index_bars["high"],
            "all_a_equal_weight_low": index_bars["low"],
            "all_a_equal_weight_close": index_bars["close"],
            "total_amount": index_bars["amount"],
            "total_volume": index_bars["volume"],
            "advancers_count": 0,
            "decliners_count": 0,
            "limit_up_count": 0,
            "limit_down_count": 0,
            "new_high_count_20": 0,
            "new_low_count_20": 0,
            "new_high_count_60": 0,
            "new_low_count_60": 0,
            "strong_stock_ratio": 0.0,
            "weak_stock_ratio": 0.0,
            "median_ret_pct": index_bars["close"] / index_bars["prev_close"] - 1,
        }
    )
    frame["scope_id"] = scope_id
    return frame
```

- [ ] Add a helper that mixes index OHLC with self-calculated breadth.

```python
def build_mixed_market_bars(index_bars: pd.DataFrame, breadth_bars: pd.DataFrame, scope_id: str) -> pd.DataFrame:
    index_market = build_market_bars_from_index(index_bars, scope_id)
    breadth_columns = [
        "date",
        "advancers_count",
        "decliners_count",
        "limit_up_count",
        "limit_down_count",
        "new_high_count_20",
        "new_low_count_20",
        "new_high_count_60",
        "new_low_count_60",
        "strong_stock_ratio",
        "weak_stock_ratio",
        "median_ret_pct",
    ]
    merged = index_market.drop(columns=[column for column in breadth_columns if column != "date"]).merge(
        breadth_bars[breadth_columns],
        on="date",
        how="left",
    )
    merged["scope_id"] = scope_id
    return merged
```

- [ ] Add tests for selected-index and mixed modes.

```python
def test_build_market_bars_from_index_uses_index_ohlc():
    index_bars = pd.DataFrame(
        [
            {
                "date": "2024-01-02",
                "index_code": "000985.CSI",
                "open": 10.0,
                "high": 11.0,
                "low": 9.0,
                "close": 10.5,
                "prev_close": 10.0,
                "volume": 1000.0,
                "amount": 2000.0,
            }
        ]
    )

    market = build_market_bars_from_index(index_bars, "000985.CSI")

    assert market.iloc[0]["all_a_equal_weight_close"] == 10.5
    assert market.iloc[0]["scope_id"] == "000985.CSI"
```

```python
def test_build_mixed_market_bars_keeps_breadth_from_self_calculated_market():
    index_bars = pd.DataFrame(
        [
            {
                "date": "2024-01-02",
                "index_code": "000985.CSI",
                "open": 10.0,
                "high": 11.0,
                "low": 9.0,
                "close": 10.5,
                "prev_close": 10.0,
                "volume": 1000.0,
                "amount": 2000.0,
            }
        ]
    )
    breadth = pd.DataFrame(
        [
            {
                "date": "2024-01-02",
                "advancers_count": 2,
                "decliners_count": 1,
                "limit_up_count": 1,
                "limit_down_count": 0,
                "new_high_count_20": 1,
                "new_low_count_20": 0,
                "new_high_count_60": 1,
                "new_low_count_60": 0,
                "strong_stock_ratio": 0.5,
                "weak_stock_ratio": 0.0,
                "median_ret_pct": 0.02,
            }
        ]
    )

    market = build_mixed_market_bars(index_bars, breadth, "000985.CSI")

    assert market.iloc[0]["all_a_equal_weight_close"] == 10.5
    assert market.iloc[0]["advancers_count"] == 2
```

- [ ] Run focused tests.

```bash
python -m pytest tests/test_market_aggregates.py -v
```

Expected: PASS.

- [ ] Commit.

```bash
git add vpa_structure_recognizer/market_aggregates.py tests/test_market_aggregates.py
git commit -m "Support index-backed market bars" -m "Market aggregation can now use local all-A breadth, selected index OHLC, or a mixed source without changing downstream feature logic.\n\nConfidence: medium\nScope-risk: moderate\nTested: python -m pytest tests/test_market_aggregates.py -v"
```

## Task 5: Extend Pipeline Market Mode Parameters

**Files:**
- Modify: `vpa_structure_recognizer/pipeline.py`
- Modify: `scripts/run_vpa_structure.py`
- Modify: `tests/test_pipeline_smoke.py`

- [ ] Add imports for run metadata.

```python
from datetime import datetime, timezone
from uuid import uuid4
```

- [ ] Extend `PipelineResult`.

```python
@dataclass(frozen=True)
class PipelineResult:
    output_db: Path
    report_path: Path
    table_counts: dict[str, int]
    run_id: str
    market_mode: str
    market_index_code: str | None
    market_scope_id: str
    sector_mode: str
```

- [ ] Add `market_mode` and `market_index_code` parameters to `run_pipeline(...)`.

```python
def run_pipeline(
    *,
    config_path: Path | str,
    start_date: str,
    end_date: str,
    source: str | None = None,
    output_db: Path | str | None = None,
    output_dir: Path | str | None = None,
    as_of_date: str | None = None,
    market_mode: str = "self_calculated_all_a",
    market_index_code: str | None = None,
) -> PipelineResult:
```

- [ ] Select market bars before feature computation.

```python
self_market_bars = build_market_bars(stock_bars)
market_scope_id = "ALL_A"
if market_mode == "self_calculated_all_a":
    market_bars = self_market_bars
elif market_mode == "selected_index":
    if not market_index_code:
        raise ValueError("market_index_code is required for selected_index mode")
    index_bars = source_reader.fetch_index_bars(market_index_code, start_date, end_date)
    if index_bars.empty:
        raise ValueError(f"No index bars found for {market_index_code}")
    market_bars = build_market_bars_from_index(index_bars, market_index_code)
    market_scope_id = market_index_code
elif market_mode == "mixed_index_breadth":
    if not market_index_code:
        raise ValueError("market_index_code is required for mixed_index_breadth mode")
    index_bars = source_reader.fetch_index_bars(market_index_code, start_date, end_date)
    if index_bars.empty:
        raise ValueError(f"No index bars found for {market_index_code}")
    market_bars = build_mixed_market_bars(index_bars, self_market_bars, market_index_code)
    market_scope_id = market_index_code
else:
    raise ValueError(f"Unsupported market_mode: {market_mode}")
```

- [ ] Pass `market_scope_id` into `_market_feature_input(...)`.

```python
compute_features(
    _market_feature_input(market_bars),
    config.windows,
    "market",
    scope_id=market_scope_id,
)
```

- [ ] Insert run metadata after table writes.

```python
report_date = (as_of_date or end_date).replace("-", "")
report_path = report_dir / f"vpa_structure_report_{report_date}.xlsx"
run_id = f"vpa-{report_date}-{uuid4().hex[:8]}"
```

- [ ] Insert run metadata after `report_date` and `report_path` are defined and before closing the DuckDB connection.

```python
insert_run_metadata(
    con,
    {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "start_date": start_date,
        "end_date": end_date,
        "as_of_date": as_of_date,
        "source": source,
        "output_db": str(output_db_path),
        "report_path": str(report_path),
        "market_mode": market_mode,
        "market_index_code": market_index_code,
        "market_scope_id": market_scope_id,
        "sector_mode": "local_industry_aggregate",
    },
)
```

- [ ] Add CLI arguments.

```python
parser.add_argument("--market-mode", default="self_calculated_all_a")
parser.add_argument("--market-index-code")
```

- [ ] Print market metadata in CLI output.

```python
print(f"run_id={result.run_id}")
print(f"market_mode={result.market_mode}")
print(f"market_index_code={result.market_index_code}")
print(f"market_scope_id={result.market_scope_id}")
print(f"sector_mode={result.sector_mode}")
```

- [ ] Add a smoke test for selected index mode using fixture `index_daily_bar_pit`.

```python
def test_pipeline_supports_selected_market_index(tmp_path):
    source_db = _build_fixture_source(tmp_path)
    result = run_pipeline(
        config_path="config/default.toml",
        start_date="2024-01-01",
        end_date="2024-01-31",
        source=str(source_db),
        output_db=tmp_path / "vpa.duckdb",
        output_dir=tmp_path,
        market_mode="selected_index",
        market_index_code="000985.CSI",
    )
    assert result.market_mode == "selected_index"
    assert result.market_scope_id == "000985.CSI"
```

- [ ] Add a smoke test for unavailable planned index candidates.

```python
def test_pipeline_rejects_planned_index_without_local_bars(tmp_path):
    source_db = _build_fixture_source(tmp_path)
    with pytest.raises(ValueError, match="No index bars found for 700081.TI"):
        run_pipeline(
            config_path="config/default.toml",
            start_date="2024-01-01",
            end_date="2024-01-31",
            source=str(source_db),
            output_db=tmp_path / "vpa.duckdb",
            output_dir=tmp_path,
            market_mode="selected_index",
            market_index_code="700081.TI",
        )
```

- [ ] Run focused tests.

```bash
python -m pytest tests/test_pipeline_smoke.py -v
```

Expected: PASS.

- [ ] Commit.

```bash
git add vpa_structure_recognizer/pipeline.py scripts/run_vpa_structure.py tests/test_pipeline_smoke.py
git commit -m "Let VPA runs choose their market scope" -m "Pipeline and CLI now support self-calculated, selected-index, and mixed market modes with metadata recorded for each run.\n\nConfidence: medium\nScope-risk: broad\nTested: python -m pytest tests/test_pipeline_smoke.py -v"
```

## Task 6: Create App Service Validation and Result API

**Files:**
- Create: `vpa_structure_recognizer/app_service.py`
- Create: `tests/test_app_service.py`

- [ ] Create service dataclasses.

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import duckdb
import pandas as pd

from vpa_structure_recognizer.pipeline import PipelineResult, run_pipeline


@dataclass(frozen=True)
class RunRequest:
    config_path: Path
    start_date: str
    end_date: str
    as_of_date: str | None
    source: str | None
    output_db: Path
    output_dir: Path
    market_mode: str
    market_index_code: str | None


@dataclass(frozen=True)
class RunSummary:
    result: PipelineResult
    metadata: dict[str, object]
```

- [ ] Add date/path/index validation.

```python
def validate_run_request(request: RunRequest) -> None:
    if request.start_date > request.end_date:
        raise ValueError("开始日期不能晚于结束日期")
    if not request.config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {request.config_path}")
    if request.source and request.source not in {"research_source", "audited_stock"}:
        source_path = Path(request.source)
        if not source_path.exists():
            raise FileNotFoundError(f"数据源不存在: {source_path}")
    if request.market_mode in {"selected_index", "mixed_index_breadth"} and not request.market_index_code:
        raise ValueError("指定指数口径需要 market_index_code")
```

- [ ] Add `run_analysis(...)`.

```python
def run_analysis(request: RunRequest) -> RunSummary:
    validate_run_request(request)
    result = run_pipeline(
        config_path=request.config_path,
        start_date=request.start_date,
        end_date=request.end_date,
        as_of_date=request.as_of_date,
        source=request.source,
        output_db=request.output_db,
        output_dir=request.output_dir,
        market_mode=request.market_mode,
        market_index_code=request.market_index_code,
    )
    return RunSummary(
        result=result,
        metadata={
            "output_db": str(result.output_db),
            "report_path": str(result.report_path),
            "table_counts": result.table_counts,
            "market_mode": result.market_mode,
            "market_index_code": result.market_index_code,
            "market_scope_id": result.market_scope_id,
            "sector_mode": result.sector_mode,
        },
    )
```

- [ ] Add result readers.

```python
def read_table(db_path: Path | str, table_name: str) -> pd.DataFrame:
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        return con.execute(f"select * from {table_name}").fetchdf()
    finally:
        con.close()


def read_latest_metadata(db_path: Path | str) -> pd.Series | None:
    metadata = read_table(db_path, "vpa_run_metadata")
    if metadata.empty:
        return None
    return metadata.sort_values("created_at").iloc[-1]
```

- [ ] Add tests for validation.

```python
def test_validate_run_request_rejects_reverse_dates(tmp_path):
    request = RunRequest(
        config_path=Path("config/default.toml"),
        start_date="2024-02-01",
        end_date="2024-01-01",
        as_of_date=None,
        source=None,
        output_db=tmp_path / "vpa.duckdb",
        output_dir=tmp_path,
        market_mode="self_calculated_all_a",
        market_index_code=None,
    )
    with pytest.raises(ValueError, match="开始日期不能晚于结束日期"):
        validate_run_request(request)
```

- [ ] Add tests for metadata reading using `init_vpa_db(...)` and `insert_run_metadata(...)`.

```python
def test_read_latest_metadata_returns_newest_run(tmp_path):
    db_path = tmp_path / "vpa.duckdb"
    con = init_vpa_db(db_path)
    insert_run_metadata(con, _metadata("run-1", "2026-05-28T00:00:00"))
    insert_run_metadata(con, _metadata("run-2", "2026-05-28T01:00:00"))
    con.close()

    metadata = read_latest_metadata(db_path)

    assert metadata["run_id"] == "run-2"
```

- [ ] Run focused tests.

```bash
python -m pytest tests/test_app_service.py -v
```

Expected: PASS.

- [ ] Commit.

```bash
git add vpa_structure_recognizer/app_service.py tests/test_app_service.py
git commit -m "Add service boundary for the local workbench" -m "The UI can validate run inputs, launch deterministic analysis, and read generated results through a small service API.\n\nConfidence: high\nScope-risk: moderate\nTested: python -m pytest tests/test_app_service.py -v"
```

## Task 7: Add Candidate Pool Views in Service Layer

**Files:**
- Modify: `vpa_structure_recognizer/app_service.py`
- Modify: `tests/test_app_service.py`

- [ ] Add candidate view names and filters.

```python
CANDIDATE_VIEW_NAMES = [
    "个股结构总表",
    "三层共振个股",
    "低位承接增强个股",
    "健康上涨个股",
    "高位供应风险个股",
    "放量破位风险个股",
]
```

- [ ] Implement `candidate_view(...)`.

```python
def candidate_view(states: pd.DataFrame, view_name: str) -> pd.DataFrame:
    stocks = states[states["scope_type"] == "stock"].copy()
    if view_name == "个股结构总表":
        return stocks
    if view_name == "三层共振个股":
        return stocks[stocks["final_rating"] == "A"]
    if view_name == "低位承接增强个股":
        return stocks[stocks["final_state"].isin({"LOW_LEVEL_SUPPORT", "POSSIBLE_ACCUMULATION"})]
    if view_name == "健康上涨个股":
        return stocks[stocks["final_state"] == "HEALTHY_UPTREND"]
    if view_name == "高位供应风险个股":
        return stocks[stocks["final_state"].isin({"HIGH_LEVEL_SUPPLY", "POSSIBLE_DISTRIBUTION"})]
    if view_name == "放量破位风险个股":
        risk_flags = stocks.get("risk_flags", pd.Series([""] * len(stocks))).fillna("")
        return stocks[(stocks["final_state"] == "BREAKDOWN") | risk_flags.str.contains("破位")]
    raise ValueError(f"未知候选池视图: {view_name}")
```

- [ ] Implement generic table filters.

```python
def filter_states(
    frame: pd.DataFrame,
    *,
    date: str | None = None,
    ratings: set[str] | None = None,
    states: set[str] | None = None,
    risk_text: str | None = None,
) -> pd.DataFrame:
    output = frame.copy()
    if date:
        output = output[output["date"] == date]
    if ratings:
        output = output[output["final_rating"].isin(ratings)]
    if states:
        output = output[output["final_state"].isin(states)]
    if risk_text:
        output = output[output["risk_flags"].fillna("").str.contains(risk_text)]
    return output
```

- [ ] Test Excel-equivalent filters.

```python
def test_candidate_view_matches_healthy_uptrend_filter():
    states = pd.DataFrame(
        [
            {"scope_type": "stock", "scope_id": "000001.SZ", "final_state": "HEALTHY_UPTREND", "final_rating": "B", "risk_flags": ""},
            {"scope_type": "stock", "scope_id": "000002.SZ", "final_state": "BREAKDOWN", "final_rating": "D", "risk_flags": "破位"},
        ]
    )

    result = candidate_view(states, "健康上涨个股")

    assert result["scope_id"].tolist() == ["000001.SZ"]
```

- [ ] Run focused tests.

```bash
python -m pytest tests/test_app_service.py -v
```

Expected: PASS.

- [ ] Commit.

```bash
git add vpa_structure_recognizer/app_service.py tests/test_app_service.py
git commit -m "Expose Excel-equivalent candidate views" -m "The workbench can browse the same stock pools as the generated workbook without duplicating screening rules in the UI.\n\nConfidence: high\nScope-risk: narrow\nTested: python -m pytest tests/test_app_service.py -v"
```

## Task 8: Add Stock Detail and Deterministic Explanation Service

**Files:**
- Modify: `vpa_structure_recognizer/report_formatter.py`
- Modify: `vpa_structure_recognizer/app_service.py`
- Modify: `tests/test_report_formatter.py`
- Modify: `tests/test_app_service.py`

- [ ] Add `format_stock_summary(...)` that accepts rows already read from DuckDB.

```python
def format_stock_summary(stock: pd.Series, labels: pd.DataFrame, sequences: pd.DataFrame) -> str:
    latest_labels = labels.sort_values(["window_n", "date"]).groupby("window_n").tail(1)
    window_states = {
        10: stock.get("state_10", "UNKNOWN"),
        20: stock.get("state_20", "UNKNOWN"),
        30: stock.get("state_30", "UNKNOWN"),
        60: stock.get("state_60", "UNKNOWN"),
        120: stock.get("state_120", "UNKNOWN"),
        240: stock.get("state_240", "UNKNOWN"),
    }
    label_text = "，".join(
        f"{int(row.window_n)}日{row.raw_label}" for row in latest_labels.itertuples(index=False)
    )
    return "\n".join(
        [
            f"股票代码：{stock.get('scope_id', '')}",
            f"最终评级：{stock.get('final_rating', '')}",
            f"当前结构：{stock.get('final_state', '')}",
            f"趋势背景：{stock.get('trend_background', '')}",
            f"位置背景：{stock.get('position_background', '')}",
            f"多窗口状态：10日{window_states[10]}，20日{window_states[20]}，30日{window_states[30]}，60日{window_states[60]}，120日{window_states[120]}，240日{window_states[240]}",
            f"主要特征：{stock.get('main_features', '')}",
            f"风险标记：{stock.get('risk_flags', '') or '未见突出风险'}",
            f"近期标签：{label_text}",
            f"看强确认：{stock.get('bullish_confirm_condition', '')}",
            f"看弱否定：{stock.get('bearish_invalidate_condition', '')}",
        ]
    )
```

- [ ] Add `stock_detail(...)` in `app_service.py`.

```python
def stock_detail(db_path: Path | str, code: str) -> dict[str, object]:
    states = read_table(db_path, "vpa_structure_state")
    labels = read_table(db_path, "vpa_bar_context_labels")
    sequences = read_table(db_path, "vpa_sequence_stats")
    stock_rows = states[(states["scope_type"] == "stock") & (states["scope_id"] == code)]
    if stock_rows.empty:
        raise ValueError(f"未找到个股: {code}")
    stock = stock_rows.sort_values("date").iloc[-1]
    stock_labels = labels[(labels["scope_type"] == "stock") & (labels["scope_id"] == code)]
    stock_sequences = sequences[(sequences["scope_type"] == "stock") & (sequences["scope_id"] == code)]
    return {
        "stock": stock,
        "labels": stock_labels,
        "sequences": stock_sequences,
        "summary": format_stock_summary(stock, stock_labels, stock_sequences),
    }
```

- [ ] Test deterministic summary.

```python
def test_format_stock_summary_uses_only_supplied_fields():
    stock = pd.Series(
        {
            "scope_id": "000001.SZ",
            "final_rating": "A",
            "final_state": "HEALTHY_UPTREND",
            "trend_background": "UPTREND",
            "position_background": "MID_HIGH",
            "state_10": "HEALTHY_UPTREND",
            "state_20": "HEALTHY_UPTREND",
            "state_30": "HEALTHY_UPTREND",
            "state_60": "UPTREND",
            "state_120": "UPTREND",
            "state_240": "UPTREND",
            "main_features": "上涨有量",
            "risk_flags": "",
            "bullish_confirm_condition": "缩量回踩不破",
            "bearish_invalidate_condition": "跌破近期低点",
        }
    )
    labels = pd.DataFrame([{"window_n": 10, "date": "2024-01-31", "raw_label": "NORMAL_UP_CONFIRM"}])

    text = format_stock_summary(stock, labels, pd.DataFrame())

    assert "000001.SZ" in text
    assert "NORMAL_UP_CONFIRM" in text
    assert "交易建议" not in text
```

- [ ] Run focused tests.

```bash
python -m pytest tests/test_report_formatter.py tests/test_app_service.py -v
```

Expected: PASS.

- [ ] Commit.

```bash
git add vpa_structure_recognizer/report_formatter.py vpa_structure_recognizer/app_service.py tests/test_report_formatter.py tests/test_app_service.py
git commit -m "Add deterministic stock detail summaries" -m "Stock explanations are generated from existing structural fields and labels without introducing LLM judgment.\n\nConfidence: high\nScope-risk: narrow\nTested: python -m pytest tests/test_report_formatter.py tests/test_app_service.py -v"
```

## Task 9: Build Streamlit App Shell

**Files:**
- Create: `apps/local_vpa_ui.py`

- [ ] Create the app shell with tabs.

```python
from __future__ import annotations

from pathlib import Path

import streamlit as st

from vpa_structure_recognizer import app_service


st.set_page_config(page_title="VPA 本地工作台", layout="wide")
st.title("VPA 本地工作台")

tabs = st.tabs(["运行分析", "结果总览", "候选池", "个股详情", "报告导出"])

with tabs[0]:
    st.subheader("运行分析")
    st.info("从本页执行确定性的量价结构分析。")

with tabs[1]:
    st.subheader("结果总览")

with tabs[2]:
    st.subheader("候选池")

with tabs[3]:
    st.subheader("个股详情")

with tabs[4]:
    st.subheader("报告导出")
```

- [ ] Start the app.

```bash
streamlit run apps/local_vpa_ui.py --server.headless true
```

Expected: Streamlit starts and prints a local URL.

- [ ] Commit.

```bash
git add apps/local_vpa_ui.py
git commit -m "Create the local VPA workbench shell" -m "The Streamlit app starts with the planned workflow tabs and no analysis logic embedded in the UI.\n\nConfidence: high\nScope-risk: narrow\nTested: streamlit run apps/local_vpa_ui.py --server.headless true"
```

## Task 10: Implement Run Analysis Tab

**Files:**
- Modify: `apps/local_vpa_ui.py`

- [ ] Add form controls for config, source, date range, output paths, and market mode.

```python
with tabs[0]:
    st.subheader("运行分析")
    with st.form("run-vpa"):
        config_path = Path(st.text_input("配置文件", "config/default.toml"))
        source_mode = st.selectbox("数据源", ["research_source", "audited_stock", "自定义 DuckDB"])
        custom_source = st.text_input("自定义数据源路径", "")
        start_date = st.date_input("开始日期")
        end_date = st.date_input("结束日期")
        as_of_date = st.date_input("报告日期", value=end_date)
        output_db = Path(st.text_input("输出 DuckDB", "outputs/vpa.duckdb"))
        output_dir = Path(st.text_input("报告目录", "outputs/reports"))
        market_mode = st.selectbox(
            "全市场判断口径",
            ["self_calculated_all_a", "selected_index", "mixed_index_breadth"],
        )
        market_index_code = st.text_input("市场指数代码", "000985.CSI")
        submitted = st.form_submit_button("执行分析")
```

- [ ] Call `app_service.run_analysis(...)` on submit.

```python
if submitted:
    source = custom_source if source_mode == "自定义 DuckDB" else source_mode
    request = app_service.RunRequest(
        config_path=config_path,
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
        as_of_date=as_of_date.isoformat(),
        source=source,
        output_db=output_db,
        output_dir=output_dir,
        market_mode=market_mode,
        market_index_code=market_index_code if market_mode != "self_calculated_all_a" else None,
    )
    try:
        summary = app_service.run_analysis(request)
    except Exception as exc:
        st.error(str(exc))
    else:
        st.success("分析完成")
        st.json(summary.metadata)
```

- [ ] Start the app and manually submit a small fixture-compatible date range if local source is available.

```bash
streamlit run apps/local_vpa_ui.py --server.headless true
```

Expected: tab renders; successful run shows output metadata or validation errors show in Chinese.

- [ ] Commit.

```bash
git add apps/local_vpa_ui.py
git commit -m "Wire the run-analysis UI to the service layer" -m "The local workbench can execute deterministic pipeline runs through app_service instead of shelling out to the CLI.\n\nConfidence: medium\nScope-risk: moderate\nTested: streamlit run apps/local_vpa_ui.py --server.headless true"
```

## Task 11: Implement Result Overview Tab

**Files:**
- Modify: `apps/local_vpa_ui.py`
- Modify: `vpa_structure_recognizer/app_service.py`
- Modify: `tests/test_app_service.py`

- [ ] Add `overview_metrics(...)` in service.

```python
def overview_metrics(db_path: Path | str) -> dict[str, object]:
    states = read_table(db_path, "vpa_structure_state")
    metadata = read_latest_metadata(db_path)
    latest_date = states["date"].max() if not states.empty else None
    latest = states[states["date"] == latest_date] if latest_date else states
    stocks = latest[latest["scope_type"] == "stock"]
    sectors = latest[latest["scope_type"] == "sector"]
    market = latest[latest["scope_type"] == "market"]
    return {
        "metadata": metadata.to_dict() if metadata is not None else {},
        "latest_date": latest_date,
        "market": market,
        "rating_counts": stocks["final_rating"].value_counts().to_dict(),
        "state_counts": stocks["final_state"].value_counts().to_dict(),
        "top_sectors": sectors.sort_values("sector_score", ascending=False).head(20),
    }
```

- [ ] Add Streamlit DB path input and metrics.

```python
with tabs[1]:
    st.subheader("结果总览")
    db_path = Path(st.text_input("结果 DuckDB", "outputs/vpa.duckdb", key="overview-db"))
    if st.button("加载总览"):
        try:
            overview = app_service.overview_metrics(db_path)
        except Exception as exc:
            st.error(str(exc))
        else:
            st.write("全市场口径", overview["metadata"].get("market_mode", "unknown"))
            st.write("板块口径", overview["metadata"].get("sector_mode", "local_industry_aggregate"))
            st.write("最近日期", overview["latest_date"])
            st.dataframe(overview["market"], use_container_width=True)
            st.bar_chart(overview["rating_counts"])
            st.bar_chart(overview["state_counts"])
            st.dataframe(overview["top_sectors"], use_container_width=True)
```

- [ ] Test metrics with a tiny DuckDB fixture.

```python
def test_overview_metrics_returns_latest_rating_counts(tmp_path):
    db_path = _build_result_db(tmp_path)

    overview = overview_metrics(db_path)

    assert overview["latest_date"] == "2024-01-31"
    assert overview["rating_counts"]["A"] == 1
```

- [ ] Run focused tests and UI smoke.

```bash
python -m pytest tests/test_app_service.py -v
streamlit run apps/local_vpa_ui.py --server.headless true
```

Expected: tests pass; app starts.

- [ ] Commit.

```bash
git add apps/local_vpa_ui.py vpa_structure_recognizer/app_service.py tests/test_app_service.py
git commit -m "Add result overview metrics to the workbench" -m "The UI can load a generated VPA database and show market scope, sector scope, market row, rating distribution, state distribution, and top sectors.\n\nConfidence: medium\nScope-risk: moderate\nTested: python -m pytest tests/test_app_service.py -v; streamlit run apps/local_vpa_ui.py --server.headless true"
```

## Task 12: Implement Candidate Pool Tab

**Files:**
- Modify: `apps/local_vpa_ui.py`

- [ ] Add candidate controls and dataframe.

```python
with tabs[2]:
    st.subheader("候选池")
    db_path = Path(st.text_input("结果 DuckDB", "outputs/vpa.duckdb", key="candidate-db"))
    view_name = st.selectbox("视图", app_service.CANDIDATE_VIEW_NAMES)
    rating_filter = set(st.multiselect("评级", ["A", "B", "C", "D", "E"]))
    risk_text = st.text_input("风险关键词", "")
    if st.button("加载候选池"):
        try:
            states = app_service.read_table(db_path, "vpa_structure_state")
            view = app_service.candidate_view(states, view_name)
            filtered = app_service.filter_states(
                view,
                ratings=rating_filter or None,
                risk_text=risk_text or None,
            )
        except Exception as exc:
            st.error(str(exc))
        else:
            st.dataframe(filtered, use_container_width=True)
            st.download_button(
                "导出 CSV",
                data=filtered.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"{view_name}.csv",
                mime="text/csv",
            )
```

- [ ] Start app and load a fixture/result DB.

```bash
streamlit run apps/local_vpa_ui.py --server.headless true
```

Expected: candidate tab loads views and offers CSV download.

- [ ] Commit.

```bash
git add apps/local_vpa_ui.py
git commit -m "Add candidate-pool browsing to the workbench" -m "Users can inspect Excel-equivalent stock pools, filter ratings and risks, and export the current view as CSV.\n\nConfidence: medium\nScope-risk: narrow\nTested: streamlit run apps/local_vpa_ui.py --server.headless true"
```

## Task 13: Implement Stock Detail Tab

**Files:**
- Modify: `apps/local_vpa_ui.py`

- [ ] Add stock-code input and details display.

```python
with tabs[3]:
    st.subheader("个股详情")
    db_path = Path(st.text_input("结果 DuckDB", "outputs/vpa.duckdb", key="detail-db"))
    code = st.text_input("股票代码", "000001.SZ")
    if st.button("加载个股"):
        try:
            detail = app_service.stock_detail(db_path, code)
        except Exception as exc:
            st.error(str(exc))
        else:
            stock = detail["stock"]
            cols = st.columns(4)
            cols[0].metric("评级", stock.get("final_rating", ""))
            cols[1].metric("结构", stock.get("final_state", ""))
            cols[2].metric("置信度", stock.get("confidence", ""))
            cols[3].metric("共振", stock.get("resonance_score", ""))
            st.text_area("结构解释", detail["summary"], height=260)
            st.dataframe(detail["labels"], use_container_width=True)
            st.dataframe(detail["sequences"], use_container_width=True)
```

- [ ] Start app and load a stock from a generated result DB.

```bash
streamlit run apps/local_vpa_ui.py --server.headless true
```

Expected: stock detail tab renders summary, labels, and sequence rows.

- [ ] Commit.

```bash
git add apps/local_vpa_ui.py
git commit -m "Add deterministic stock detail view" -m "The workbench can inspect one stock's rating, structure state, label details, sequence stats, and template-based explanation.\n\nConfidence: medium\nScope-risk: narrow\nTested: streamlit run apps/local_vpa_ui.py --server.headless true"
```

## Task 14: Implement Report Export Tab

**Files:**
- Modify: `apps/local_vpa_ui.py`
- Modify: `vpa_structure_recognizer/app_service.py`
- Modify: `tests/test_app_service.py`

- [ ] Add `list_output_files(...)`.

```python
def list_output_files(output_root: Path | str = "outputs") -> pd.DataFrame:
    root = Path(output_root)
    files = []
    for path in sorted(root.glob("**/*")):
        if path.is_file() and path.suffix in {".duckdb", ".xlsx", ".csv"}:
            files.append(
                {
                    "path": str(path),
                    "suffix": path.suffix,
                    "size_bytes": path.stat().st_size,
                    "modified_at": path.stat().st_mtime,
                }
            )
    return pd.DataFrame(files)
```

- [ ] Add report tab.

```python
with tabs[4]:
    st.subheader("报告导出")
    output_root = Path(st.text_input("输出目录", "outputs"))
    if st.button("刷新文件列表"):
        try:
            files = app_service.list_output_files(output_root)
        except Exception as exc:
            st.error(str(exc))
        else:
            st.dataframe(files, use_container_width=True)
```

- [ ] Test file listing.

```python
def test_list_output_files_returns_reports_and_databases(tmp_path):
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "report.xlsx").write_bytes(b"xlsx")
    (tmp_path / "vpa.duckdb").write_bytes(b"db")

    files = list_output_files(tmp_path)

    assert set(files["suffix"]) == {".xlsx", ".duckdb"}
```

- [ ] Run focused tests and UI smoke.

```bash
python -m pytest tests/test_app_service.py -v
streamlit run apps/local_vpa_ui.py --server.headless true
```

Expected: tests pass; app starts.

- [ ] Commit.

```bash
git add apps/local_vpa_ui.py vpa_structure_recognizer/app_service.py tests/test_app_service.py
git commit -m "Add output file browsing to the workbench" -m "Users can inspect generated DuckDB, Excel, and CSV files from the local reports area.\n\nConfidence: high\nScope-risk: narrow\nTested: python -m pytest tests/test_app_service.py -v; streamlit run apps/local_vpa_ui.py --server.headless true"
```

## Task 15: Update Documentation

**Files:**
- Modify: `README.md`
- Modify: `docs/superpowers/specs/2026-05-27-local-vpa-ui-design.md` if implementation decisions diverge

- [ ] Add local UI run command to `README.md`.

```markdown
Run the local workbench:

    streamlit run apps/local_vpa_ui.py
```

- [ ] Document market modes.

```markdown
Market modes:

- `self_calculated_all_a`: project-local all-A breadth and equal-weight OHLC.
- `selected_index`: uses an upstream index from `index_daily_bar_pit`.
- `mixed_index_breadth`: uses index OHLC plus project-local all-A breadth.

`700081.TI` is tracked as a planned THS candidate and becomes executable only after upstream ingestion provides local daily bars.
```

- [ ] Document current sector mode.

```markdown
The first-version sector mode is `local_industry_aggregate`: stocks are grouped by `industry_code` from `industry_classification_pit`, with OHLC averaged and volume/amount summed. It is not a Tushare SW index series unless upstream SW bars are added later.
```

- [ ] Run docs-adjacent verification.

```bash
python -m pytest tests -v
```

Expected: PASS.

- [ ] Commit.

```bash
git add README.md docs/superpowers/specs/2026-05-27-local-vpa-ui-design.md
git commit -m "Document the local VPA workbench" -m "README now explains how to run the local UI and how market and sector data scopes are interpreted.\n\nConfidence: high\nScope-risk: narrow\nTested: python -m pytest tests -v"
```

## Task 16: Final Verification

**Files:**
- No planned edits.

- [ ] Run the full test suite.

```bash
python -m pytest tests -v
```

Expected: PASS.

- [ ] Run a CLI smoke batch.

```bash
python scripts/run_vpa_structure.py \
  --config config/default.toml \
  --start-date 2024-01-01 \
  --end-date 2024-01-31 \
  --source /home/nan/alpha-find-v2/output/research_source.duckdb \
  --output-db outputs/vpa_smoke.duckdb \
  --output-dir outputs/reports
```

Expected: command prints `output_db=...`, `report_path=...`, `table_counts=...`, and market metadata.

- [ ] Run selected-index CLI smoke.

```bash
python scripts/run_vpa_structure.py \
  --config config/default.toml \
  --start-date 2024-01-01 \
  --end-date 2024-01-31 \
  --source /home/nan/alpha-find-v2/output/research_source.duckdb \
  --output-db outputs/vpa_index_smoke.duckdb \
  --output-dir outputs/reports \
  --market-mode selected_index \
  --market-index-code 000985.CSI
```

Expected: command completes and `market_scope_id=000985.CSI`.

- [ ] Run Streamlit smoke.

```bash
streamlit run apps/local_vpa_ui.py --server.headless true
```

Expected: Streamlit starts and prints a local URL.

- [ ] Inspect generated metadata table.

```bash
python - <<'PY'
import duckdb
con = duckdb.connect("outputs/vpa_index_smoke.duckdb", read_only=True)
print(con.execute("select market_mode, market_index_code, market_scope_id, sector_mode from vpa_run_metadata").fetchdf())
con.close()
PY
```

Expected: `selected_index`, `000985.CSI`, `000985.CSI`, `local_industry_aggregate`.

- [ ] Commit any final documentation or small fixes with a Lore-format commit message.

## Backlog Task A: Upstream THS Market Index Ingestion

**Owner:** upstream data project, not the local UI implementation.

**Goal:** Make `700081.TI` executable by adding verified THS index reference and daily bars to the upstream DuckDB.

**Expected local tables after upstream sync:**

- `index_basic_ref` contains `700081.TI`.
- `index_daily_bar_pit` contains `700081.TI` rows for the date range used by VPA.

**Verification command in this repo:**

```bash
python - <<'PY'
import duckdb
con = duckdb.connect("/home/nan/alpha-find-v2/output/research_source.duckdb", read_only=True)
print(con.execute("select * from index_basic_ref where ts_code='700081.TI'").fetchdf())
print(con.execute("""
select index_code, min(trade_date), max(trade_date), count(*)
from index_daily_bar_pit
where index_code='700081.TI'
group by index_code
""").fetchdf())
con.close()
PY
```

Expected: both queries return non-empty results.

## Backlog Task B: Upstream SW Industry Index Ingestion

**Owner:** upstream data project, not the first-version local UI implementation.

**Goal:** Add formal申万 industry index bars so sector analysis can choose between local industry aggregation and upstream SW index K lines.

**Tushare interfaces to sync upstream:**

- `index_basic(market='SW')`
- `index_classify(src='SW2021')`
- `index_member_all(...)`
- `sw_daily(...)`

**Expected local behavior after upstream sync:**

- `app_service` can list available SW sector indexes.
- Pipeline can use SW index OHLC for sector trend.
- Mixed sector mode can combine SW index trend with local member breadth.

## Self-Review Checklist

- [x] Every first-version spec requirement maps to a task:
  - Run analysis UI: Tasks 6, 9, 10.
  - Result browsing: Tasks 6, 11, 14.
  - Candidate pools: Task 7, Task 12.
  - Stock detail: Task 8, Task 13.
  - Market scope selection: Tasks 2, 3, 4, 5, 10, 11, 16.
  - Sector source visibility: Tasks 2, 6, 11, 15.
  - LLM exclusion: Task 8 and documentation in Task 15.
- [x] Backlog data-source requirements are separated from first-version UI implementation.
- [x] No task writes to upstream DuckDB files.
- [x] Every task has file paths and verification commands.
- [x] Frequent commits are included after each implementation slice.
