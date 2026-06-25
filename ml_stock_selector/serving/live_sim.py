from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import shutil

import duckdb
import pandas as pd

from ml_stock_selector.backtest.execution import ExecutionConfig, simulate_rebalance_orders
from ml_stock_selector.portfolio.allocator import allocate_weights
from ml_stock_selector.portfolio.constraints import PortfolioConstraints
from ml_stock_selector.portfolio.constructor import TARGET_COLUMNS, construct_portfolio_targets_v2
from ml_stock_selector.portfolio.holding_policy import HoldingPolicy
from ml_stock_selector.storage import upsert_dataframe

SCORE_VERSION = "preferred_adv10m_fulladv015_top12"
PROFIT_PROTECT_RUN_ID = "wf_v2_ret5_fund_fixed_a160_r120_20260621"
PROFIT_PROTECT_SCORE_VERSION = "v2_alpha_ret5d_fund_fixed_a160_r120_20260621"
PROFIT_PROTECT_PORTFOLIO_ID = "mkt_tier_profit_protect"
PROFIT_PROTECT_MANIFEST_ROOT = Path("outputs/ml/cache/folds_ret5_fundamental_fixed_rounds_20260621")
PROFIT_PROTECT_SERVING_FOLD_ID = "wf_2026"


@dataclass(frozen=True)
class LiveSimConfig:
    account_id: str = "preferred_adv10m_paper"
    initial_cash: float = 300_000.0
    portfolio_id: str = "preferred_adv10m_fulladv015_top12"
    score_version: str = SCORE_VERSION
    target_positions: int = 12
    report_dir: Path = Path("outputs/ml/live_sim/reports")
    profit_protect_enabled: bool = False
    profit_protect_min_days: int = 3
    profit_protect_min_gain: float = 0.03
    profit_protect_exit_below: float = 0.005
    market_zero_below: float | None = None
    market_half_below: float | None = None
    market_min_adv20_amount: float = 10_000_000.0
    execution: ExecutionConfig = ExecutionConfig(
        slippage_bps=5.0,
        commission_bps=3.0,
        stamp_duty_bps=5.0,
        allow_fractional_shares=False,
    )
    constraints: PortfolioConstraints = PortfolioConstraints(
        target_positions=12,
        hard_max_positions=15,
        max_initial_entries=12,
        max_new_entries_per_day=4,
        min_adv20_amount=10_000_000.0,
        candidate_min_trade_score=0.75,
        core_min_trade_score=0.75,
        candidate_absolute_min_rank_pct=0.70,
        candidate_active_min_rank_pct=0.70,
        candidate_risk_max_rank_pct=0.65,
        core_absolute_min_rank_pct=0.75,
        core_active_min_rank_pct=0.65,
        core_risk_max_rank_pct=0.55,
        exclude_bse=True,
        holding_policy=HoldingPolicy(
            min_hold_days=3,
            target_hold_days=5,
            max_hold_days=10,
            sell_score_threshold=0.45,
            risk_exit_rank_pct=0.85,
            risk_exit_prob=0.70,
            sell_if_not_candidate_after_target_days=True,
            force_exit_after_max_hold_days=True,
            allow_score_exit_before_min_hold=False,
        ),
    )


def profit_protect_live_sim_config(
    *,
    account_id: str = "profit_protect_paper",
    initial_cash: float = 1_000_000.0,
    report_dir: Path = Path("outputs/ml/live_sim/reports"),
) -> LiveSimConfig:
    return LiveSimConfig(
        account_id=account_id,
        initial_cash=initial_cash,
        portfolio_id=PROFIT_PROTECT_PORTFOLIO_ID,
        score_version=PROFIT_PROTECT_SCORE_VERSION,
        target_positions=12,
        report_dir=report_dir,
        profit_protect_enabled=True,
        profit_protect_min_days=3,
        profit_protect_min_gain=0.03,
        profit_protect_exit_below=0.005,
        market_zero_below=0.375,
        market_half_below=0.475,
        market_min_adv20_amount=10_000_000.0,
        execution=ExecutionConfig(
            slippage_bps=10.0,
            commission_bps=3.0,
            stamp_duty_bps=5.0,
            allow_fractional_shares=False,
        ),
        constraints=PortfolioConstraints(
            target_positions=12,
            hard_max_positions=15,
            max_initial_entries=12,
            max_new_entries_per_day=4,
            min_adv20_amount=10_000_000.0,
            candidate_min_trade_score=0.75,
            core_min_trade_score=0.75,
            candidate_absolute_min_rank_pct=0.70,
            candidate_active_min_rank_pct=0.70,
            candidate_risk_max_rank_pct=0.55,
            core_absolute_min_rank_pct=0.75,
            core_active_min_rank_pct=0.65,
            core_risk_max_rank_pct=0.45,
            exclude_bse=True,
            holding_policy=HoldingPolicy(
                min_hold_days=3,
                target_hold_days=5,
                max_hold_days=10,
                sell_score_threshold=0.35,
                risk_exit_rank_pct=0.75,
                risk_exit_prob=0.60,
                sell_if_not_candidate_after_target_days=True,
                force_exit_after_max_hold_days=True,
                allow_score_exit_before_min_hold=False,
            ),
        ),
    )


@dataclass(frozen=True)
class LiveSimDayResult:
    account_id: str
    plan_date: str
    execution_date: str
    planned_orders: pd.DataFrame
    executions: pd.DataFrame
    holdings: pd.DataFrame
    nav: dict[str, float | str]
    report_path: Path | None = None


def live_sim_reproducibility_snapshot(config: LiveSimConfig) -> dict[str, object]:
    constraints = asdict(config.constraints)
    execution = asdict(config.execution)
    return {
        "score_version": config.score_version,
        "account_id": config.account_id,
        "portfolio_id": config.portfolio_id,
        "initial_cash": config.initial_cash,
        "target_positions": config.target_positions,
        "profit_protect_enabled": config.profit_protect_enabled,
        "profit_protect_min_days": config.profit_protect_min_days,
        "profit_protect_min_gain": config.profit_protect_min_gain,
        "profit_protect_exit_below": config.profit_protect_exit_below,
        "market_zero_below": config.market_zero_below,
        "market_half_below": config.market_half_below,
        "execution": execution,
        "constraints": constraints,
    }


def init_live_sim_db(path: Path | str) -> duckdb.DuckDBPyConnection:
    db_path = Path(path)
    if str(path) != ":memory:":
        db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(path))
    con.execute(
        """
        create table if not exists live_sim_account (
            account_id varchar primary key,
            initial_cash double not null,
            created_at varchar not null
        )
        """
    )
    con.execute(
        """
        create table if not exists live_sim_planned_orders (
            account_id varchar not null,
            decision_date varchar not null,
            execution_date varchar not null,
            code varchar not null,
            side varchar not null,
            target_weight double not null,
            trade_score_v2 double,
            absolute_rank_pct double,
            active_rank_pct double,
            risk_rank_pct double,
            adv20_amount double,
            estimated_price double,
            estimated_qty double,
            target_value double,
            entry_reason varchar,
            signal_action varchar,
            status varchar not null,
            generated_at varchar not null,
            primary key (account_id, decision_date, code)
        )
        """
    )
    for column_sql in [
        "alter table live_sim_planned_orders add column if not exists estimated_price double",
        "alter table live_sim_planned_orders add column if not exists estimated_qty double",
        "alter table live_sim_planned_orders add column if not exists target_value double",
        "alter table live_sim_planned_orders add column if not exists portfolio_id varchar",
        "alter table live_sim_planned_orders add column if not exists hold_reason varchar",
        "alter table live_sim_planned_orders add column if not exists exit_reason varchar",
        "alter table live_sim_planned_orders add column if not exists sell_blocked_reason varchar",
        "alter table live_sim_planned_orders add column if not exists target_exposure_scalar double",
    ]:
        con.execute(column_sql)
    con.execute(
        """
        create table if not exists live_sim_executions (
            account_id varchar not null,
            decision_date varchar not null,
            sim_date varchar not null,
            code varchar not null,
            side varchar not null,
            qty double,
            target_weight double,
            fill_px double,
            commission double,
            stamp_duty double,
            fees double,
            status varchar,
            reason varchar,
            realized_pnl double,
            generated_at varchar not null,
            primary key (account_id, decision_date, sim_date, code, side)
        )
        """
    )
    for column_sql in [
        "alter table live_sim_executions add column if not exists commission double",
        "alter table live_sim_executions add column if not exists stamp_duty double",
        "alter table live_sim_executions add column if not exists fees double",
        "alter table live_sim_executions add column if not exists entry_date varchar",
        "alter table live_sim_executions add column if not exists entry_price double",
        "alter table live_sim_executions add column if not exists exit_date varchar",
        "alter table live_sim_executions add column if not exists holding_days integer",
        "alter table live_sim_executions add column if not exists entry_trade_score double",
        "alter table live_sim_executions add column if not exists exit_trade_score double",
        "alter table live_sim_executions add column if not exists entry_reason varchar",
        "alter table live_sim_executions add column if not exists exit_reason varchar",
        "alter table live_sim_executions add column if not exists sell_blocked_reason varchar",
        "alter table live_sim_executions add column if not exists strategy_id varchar",
    ]:
        con.execute(column_sql)
    con.execute(
        """
        create table if not exists live_sim_holdings (
            account_id varchar not null,
            code varchar not null,
            qty double not null,
            entry_date varchar,
            entry_price double,
            entry_trade_score double,
            entry_reason varchar,
            updated_at varchar not null,
            primary key (account_id, code)
        )
        """
    )
    con.execute(
        """
        create table if not exists live_sim_nav (
            account_id varchar not null,
            sim_date varchar not null,
            nav double not null,
            cash double not null,
            holding_market_value double not null,
            total_return double not null,
            max_drawdown double not null,
            primary key (account_id, sim_date)
        )
        """
    )
    con.execute(
        """
        create table if not exists live_sim_holding_path_stats (
            account_id varchar not null,
            code varchar not null,
            max_close_ret double,
            max_high_ret double,
            current_close_ret double,
            min_low_ret double,
            updated_at varchar not null,
            primary key (account_id, code)
        )
        """
    )
    con.execute(
        """
        create table if not exists live_model_bundle (
            bundle_id varchar primary key,
            strategy_id varchar not null,
            score_version varchar not null,
            source_run_id varchar not null,
            source_fold_id varchar,
            source_manifest_path varchar,
            source_manifest_hash varchar,
            alpha_model_id varchar,
            risk_model_id varchar,
            alpha_artifact_uri varchar not null,
            risk_artifact_uri varchar not null,
            feature_schema_uri varchar not null,
            feature_set_id varchar,
            label_base varchar,
            horizon_d integer,
            train_window_mode varchar,
            source_train_window_mode varchar,
            alpha_rounds integer,
            risk_rounds integer,
            activated_at varchar,
            deactivated_at varchar,
            is_active boolean not null,
            notes varchar
        )
        """
    )
    con.execute(
        """
        create table if not exists live_strategy_config_snapshot (
            snapshot_id varchar primary key,
            strategy_id varchar not null,
            account_id varchar not null,
            score_version varchar not null,
            config_json varchar not null,
            created_at varchar not null
        )
        """
    )
    con.execute(
        """
        create table if not exists live_predictions_daily (
            trade_date varchar not null,
            code varchar not null,
            bundle_id varchar not null,
            strategy_id varchar not null,
            score_version varchar not null,
            model_id varchar,
            horizon_d integer,
            absolute_score double,
            absolute_rank_pct double,
            active_score double,
            active_rank_pct double,
            risk_prob double,
            risk_rank_pct double,
            trade_score_v2 double,
            adv20_amount double,
            generated_at varchar,
            primary key (trade_date, code, bundle_id)
        )
        """
    )
    return con


def activate_profit_protect_live_bundle(
    con: duckdb.DuckDBPyConnection,
    *,
    manifest_root: Path | str = PROFIT_PROTECT_MANIFEST_ROOT,
    fold_id: str = PROFIT_PROTECT_SERVING_FOLD_ID,
    bundle_id: str | None = None,
    artifact_snapshot_dir: Path | str | None = None,
) -> dict[str, object]:
    manifest_path = Path(manifest_root) / f"run_id={PROFIT_PROTECT_RUN_ID}" / f"fold_id={fold_id}" / "manifest.json"
    manifest = _read_manifest(manifest_path)
    artifacts = dict(manifest.get("artifacts") or {})
    alpha = dict(artifacts["absolute"])
    risk = dict(artifacts["risk"])
    alpha_artifact_uri = str(alpha.get("artifact_uri"))
    risk_artifact_uri = str(risk.get("artifact_uri"))
    feature_schema_uri = str(alpha.get("feature_schema_uri") or risk.get("feature_schema_uri"))
    source_manifest_path = str(manifest_path)
    source_manifest_hash = _file_sha256(manifest_path)

    if artifact_snapshot_dir is not None:
        snapshot = _snapshot_live_model_artifacts(
            snapshot_dir=Path(artifact_snapshot_dir),
            manifest_path=manifest_path,
            alpha_artifact=Path(alpha_artifact_uri),
            risk_artifact=Path(risk_artifact_uri),
            feature_schema=Path(feature_schema_uri),
        )
        alpha_artifact_uri = str(snapshot["alpha_artifact_uri"])
        risk_artifact_uri = str(snapshot["risk_artifact_uri"])
        feature_schema_uri = str(snapshot["feature_schema_uri"])
        source_manifest_path = str(snapshot["source_manifest_path"])

    bundle = {
        "bundle_id": bundle_id or f"{PROFIT_PROTECT_PORTFOLIO_ID}:{PROFIT_PROTECT_SCORE_VERSION}",
        "strategy_id": PROFIT_PROTECT_PORTFOLIO_ID,
        "score_version": PROFIT_PROTECT_SCORE_VERSION,
        "source_run_id": PROFIT_PROTECT_RUN_ID,
        "source_fold_id": manifest.get("fold_id"),
        "source_manifest_path": source_manifest_path,
        "source_manifest_hash": source_manifest_hash,
        "alpha_model_id": alpha.get("model_id"),
        "risk_model_id": risk.get("model_id"),
        "alpha_artifact_uri": alpha_artifact_uri,
        "risk_artifact_uri": risk_artifact_uri,
        "feature_schema_uri": feature_schema_uri,
        "feature_set_id": alpha.get("feature_set_id") or manifest.get("feature_set_id"),
        "label_base": alpha.get("label_base") or manifest.get("label_base"),
        "horizon_d": alpha.get("horizon_d") or manifest.get("horizon_d"),
        "train_window_mode": manifest.get("train_window_mode"),
        "source_train_window_mode": manifest.get("source_train_window_mode"),
        "alpha_rounds": manifest.get("fixed_alpha_rounds"),
        "risk_rounds": manifest.get("fixed_risk_rounds"),
        "activated_at": _now(),
        "deactivated_at": None,
        "is_active": True,
        "notes": "profit_protect active live bundle",
    }
    for required in ["alpha_artifact_uri", "risk_artifact_uri", "feature_schema_uri"]:
        path = Path(str(bundle[required]))
        if not path.exists():
            raise FileNotFoundError(path)
    _upsert_live_model_bundle(con, bundle)
    return bundle


def load_active_live_model_bundle(con: duckdb.DuckDBPyConnection, strategy_id: str) -> dict[str, object]:
    row = con.execute(
        """
        select *
        from live_model_bundle
        where strategy_id = ?
          and is_active = true
        order by activated_at desc
        limit 1
        """,
        [strategy_id],
    ).fetchdf()
    if row.empty:
        raise RuntimeError(f"no active live model bundle for strategy_id={strategy_id}")
    return row.iloc[0].to_dict()


def save_live_strategy_config_snapshot(
    con: duckdb.DuckDBPyConnection,
    config: LiveSimConfig,
    *,
    snapshot_id: str | None = None,
) -> dict[str, object]:
    created_at = _now()
    snapshot = {
        "snapshot_id": snapshot_id or f"{config.portfolio_id}:{config.account_id}:{created_at}",
        "strategy_id": config.portfolio_id,
        "account_id": config.account_id,
        "score_version": config.score_version,
        "config_json": json.dumps(live_sim_reproducibility_snapshot(config), sort_keys=True),
        "created_at": created_at,
    }
    con.execute(
        """
        insert into live_strategy_config_snapshot
        values (?, ?, ?, ?, ?, ?)
        on conflict (snapshot_id) do update set
            strategy_id = excluded.strategy_id,
            account_id = excluded.account_id,
            score_version = excluded.score_version,
            config_json = excluded.config_json,
            created_at = excluded.created_at
        """,
        [
            snapshot["snapshot_id"],
            snapshot["strategy_id"],
            snapshot["account_id"],
            snapshot["score_version"],
            snapshot["config_json"],
            snapshot["created_at"],
        ],
    )
    return snapshot


def upsert_live_predictions(
    con: duckdb.DuckDBPyConnection,
    predictions: pd.DataFrame,
    *,
    bundle_id: str,
) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    bundle = _load_live_model_bundle_by_id(con, bundle_id)
    out = predictions.copy()
    out["bundle_id"] = bundle_id
    out["strategy_id"] = bundle["strategy_id"]
    out["score_version"] = bundle["score_version"]
    if "generated_at" not in out:
        out["generated_at"] = _now()
    for column in [
        "model_id",
        "horizon_d",
        "absolute_score",
        "absolute_rank_pct",
        "active_score",
        "active_rank_pct",
        "risk_prob",
        "risk_rank_pct",
        "trade_score_v2",
        "adv20_amount",
    ]:
        if column not in out:
            out[column] = None
    keep = [
        "trade_date",
        "code",
        "bundle_id",
        "strategy_id",
        "score_version",
        "model_id",
        "horizon_d",
        "absolute_score",
        "absolute_rank_pct",
        "active_score",
        "active_rank_pct",
        "risk_prob",
        "risk_rank_pct",
        "trade_score_v2",
        "adv20_amount",
        "generated_at",
    ]
    upsert_dataframe(con, "live_predictions_daily", out[keep], ["trade_date", "code", "bundle_id"])
    return out[keep]


def load_live_predictions(
    con: duckdb.DuckDBPyConnection,
    trade_date: str,
    *,
    bundle_id: str | None = None,
    strategy_id: str | None = None,
) -> pd.DataFrame:
    clauses = ["trade_date = ?"]
    params: list[object] = [trade_date]
    if bundle_id is not None:
        clauses.append("bundle_id = ?")
        params.append(bundle_id)
    if strategy_id is not None:
        clauses.append("strategy_id = ?")
        params.append(strategy_id)
    return con.execute(
        f"""
        select *
        from live_predictions_daily
        where {' and '.join(clauses)}
        order by code
        """,
        params,
    ).fetchdf()


def archived_adv_score(predictions: pd.DataFrame, score_version: str = SCORE_VERSION) -> pd.DataFrame:
    out = predictions.copy()
    if out.empty:
        out["full_prediction_pool_adv_pct"] = pd.Series(dtype=float)
        out["trade_score_v2"] = pd.Series(dtype=float)
        out["alpha_rank_pct"] = pd.Series(dtype=float)
        out["active_rank_pct"] = pd.Series(dtype=float)
        out["core_score"] = pd.Series(dtype=float)
        out["trade_score"] = pd.Series(dtype=float)
        out["score_version"] = score_version
        return out
    out["absolute_rank_pct"] = pd.to_numeric(out["absolute_rank_pct"], errors="coerce").fillna(0.0)
    if "risk_rank_pct" not in out:
        out["risk_rank_pct"] = 0.0
    else:
        out["risk_rank_pct"] = pd.to_numeric(out["risk_rank_pct"], errors="coerce").fillna(0.0)
    if "risk_prob" not in out:
        out["risk_prob"] = 0.0
    else:
        out["risk_prob"] = pd.to_numeric(out["risk_prob"], errors="coerce").fillna(0.0)
    out["full_prediction_pool_adv_pct"] = (
        out.groupby("trade_date")["adv20_amount"].rank(method="average", pct=True, ascending=True)
        if "trade_date" in out
        else out["adv20_amount"].rank(method="average", pct=True, ascending=True)
    )
    out["trade_score_v2"] = 0.85 * out["absolute_rank_pct"].fillna(0.0) + 0.15 * (1.0 - out["full_prediction_pool_adv_pct"].fillna(1.0))
    out["alpha_rank_pct"] = out["absolute_rank_pct"]
    out["active_rank_pct"] = out["absolute_rank_pct"]
    out["core_score"] = out["trade_score_v2"]
    out["trade_score"] = out["trade_score_v2"]
    out["score_version"] = score_version
    return out.sort_values(["trade_date", "trade_score_v2", "code"], ascending=[True, False, True]).reset_index(drop=True)


def run_live_sim_day(
    con: duckdb.DuckDBPyConnection,
    as_of_date: str,
    predictions: pd.DataFrame,
    bars: pd.DataFrame,
    config: LiveSimConfig,
) -> LiveSimDayResult:
    _ensure_account(con, config)
    executions = _settle_due_orders(con, as_of_date, bars, config)
    holdings = _load_holdings(con, config.account_id)
    holdings = _refresh_holding_path_stats(con, config.account_id, holdings, bars, as_of_date)
    nav = _record_nav(con, as_of_date, bars, config, holdings)
    execution_date = _next_trading_day(as_of_date, bars)
    targets = _build_targets(predictions, holdings, config, as_of_date, _sim_trading_dates(con, config.account_id, as_of_date), bars)
    planned = _plan_orders(con, as_of_date, execution_date, targets, holdings, bars, float(nav["nav"]), config)
    holdings_report = _annotate_holdings_for_report(holdings, bars, as_of_date)
    result = LiveSimDayResult(config.account_id, as_of_date, execution_date, planned, executions, holdings_report, nav)
    report_path = config.report_dir / f"live_sim_summary_{as_of_date}.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(generate_markdown_report(result), encoding="utf-8")
    return LiveSimDayResult(config.account_id, as_of_date, execution_date, planned, executions, holdings_report, nav, report_path)


def generate_markdown_report(result: LiveSimDayResult) -> str:
    nav = result.nav
    lines = [
        f"# 实盘模拟日报 {result.plan_date}",
        "",
        "## 账户摘要",
        f"- 账户: {result.account_id}",
        f"- NAV: {_money(nav.get('nav', 0.0))}",
        f"- 现金: {_money(nav.get('cash', 0.0))}",
        f"- 持仓总市值: {_money(nav.get('holding_market_value', 0.0))}",
        f"- 当前收益: {_pct(nav.get('total_return', 0.0))}",
        f"- 最大回撤: {_pct(nav.get('max_drawdown', 0.0))}",
        "",
        "## 当日成交",
        _markdown_table(result.executions, ["code", "side", "qty", "fill_px", "fees", "status", "reason", "exit_reason"]),
        "",
        "## 当前持仓",
        _markdown_table(result.holdings, ["code", "qty", "current_price", "market_value", "entry_price", "entry_date", "entry_trade_score", "current_close_ret", "max_high_ret"]),
        "",
        "## 下一交易日计划",
        f"- 执行日期: {result.execution_date}",
        _markdown_table(result.planned_orders, ["code", "side", "estimated_price", "estimated_qty", "target_value", "target_weight", "trade_score_v2", "adv20_amount", "exit_reason", "status"]),
        "",
    ]
    return "\n".join(lines)


def _ensure_account(con: duckdb.DuckDBPyConnection, config: LiveSimConfig) -> None:
    now = _now()
    con.execute(
        """
        insert into live_sim_account values (?, ?, ?)
        on conflict (account_id) do nothing
        """,
        [config.account_id, config.initial_cash, now],
    )
    if con.execute("select count(*) from live_sim_nav where account_id = ?", [config.account_id]).fetchone()[0] == 0:
        con.execute(
            "insert into live_sim_nav values (?, ?, ?, ?, ?, ?, ?)",
            [config.account_id, "INITIAL", config.initial_cash, config.initial_cash, 0.0, 0.0, 0.0],
        )


def _build_targets(
    predictions: pd.DataFrame,
    holdings: pd.DataFrame,
    config: LiveSimConfig,
    as_of_date: str,
    trading_dates: list[str] | None = None,
    bars: pd.DataFrame | None = None,
) -> pd.DataFrame:
    scored = archived_adv_score(predictions, config.score_version)
    # Signal generation must not depend on next-open tradeability columns, which are future
    # information at T-day close. Settlement handles limits and pauses with actual T+1 bars.
    scored = scored.drop(columns=[col for col in ["can_buy_next_open", "can_sell_next_open"] if col in scored], errors="ignore")
    constructor_holdings = _holdings_for_constructor(holdings, as_of_date, trading_dates)
    targets = construct_portfolio_targets_v2(
        scored,
        config.constraints,
        config.portfolio_id,
        current_holdings=constructor_holdings,
        score_version=config.score_version,
    )
    targets = _apply_profit_protection(targets, constructor_holdings, config, as_of_date)
    if targets.empty:
        return targets
    weighted = allocate_weights(targets, 1.0 / config.target_positions, 1.0 / config.target_positions, allow_cash=True)
    exposure_scalar = _market_exposure_scalar(bars if bars is not None else pd.DataFrame(), as_of_date, config)
    weighted = _scale_target_exposure(weighted, exposure_scalar)
    weighted["target_exposure_scalar"] = exposure_scalar
    enrich_cols = ["trade_date", "code", "trade_score_v2", "absolute_rank_pct", "active_rank_pct", "risk_rank_pct", "adv20_amount"]
    available = [col for col in enrich_cols if col in scored.columns]
    if available:
        weighted = weighted.merge(scored[available].drop_duplicates(["trade_date", "code"]), on=["trade_date", "code"], how="left", suffixes=("", "_candidate"))
        for col in ["trade_score_v2", "absolute_rank_pct", "active_rank_pct", "risk_rank_pct", "adv20_amount"]:
            candidate_col = f"{col}_candidate"
            if candidate_col in weighted:
                if col not in weighted:
                    weighted[col] = weighted[candidate_col]
                else:
                    weighted[col] = weighted[col].fillna(weighted[candidate_col])
                weighted = weighted.drop(columns=[candidate_col])
    return weighted


def _plan_orders(
    con: duckdb.DuckDBPyConnection,
    decision_date: str,
    execution_date: str,
    targets: pd.DataFrame,
    holdings: pd.DataFrame,
    bars: pd.DataFrame,
    nav: float,
    config: LiveSimConfig,
) -> pd.DataFrame:
    existing = con.execute(
        "select * from live_sim_planned_orders where account_id = ? and decision_date = ? order by code",
        [config.account_id, decision_date],
    ).fetchdf()
    if not existing.empty:
        estimate_cols = [col for col in ["estimated_price", "estimated_qty", "target_value", "adv20_amount"] if col in existing.columns]
        if estimate_cols and not existing[estimate_cols].isna().all().any():
            return existing
        con.execute(
            "delete from live_sim_planned_orders where account_id = ? and decision_date = ? and status = 'planned'",
            [config.account_id, decision_date],
        )
    current_codes = set(holdings["code"].astype(str)) if not holdings.empty else set()
    target_codes = set(targets["code"].astype(str)) if not targets.empty and "code" in targets else set()
    rows = []
    target_by_code = _rows_by_code(targets)
    close_prices = _latest_close_prices(bars, decision_date)
    now = _now()
    for code in sorted(current_codes | target_codes):
        target = target_by_code.get(code)
        target_weight = float(target.get("target_weight", 0.0)) if target is not None else 0.0
        signal_action = str(_get(target, "signal_action") or "").lower() if target is not None else ""
        if code not in current_codes and target_weight <= 0.0:
            continue
        side = (
            "sell"
            if signal_action == "sell" or code not in target_codes
            else "buy"
            if code not in current_codes
            else "hold"
        )
        if side == "hold":
            continue
        estimated_price = close_prices.get(code)
        target_value = nav * target_weight
        estimated_qty = _estimated_qty(side, target_value, estimated_price, holdings, code, config)
        rows.append(
            {
                "account_id": config.account_id,
                "decision_date": decision_date,
                "execution_date": execution_date,
                "code": code,
                "side": side,
                "target_weight": target_weight,
                "trade_score_v2": _get(target, "trade_score", "trade_score_v2"),
                "absolute_rank_pct": _get(target, "absolute_rank_pct"),
                "active_rank_pct": _get(target, "active_rank_pct"),
                "risk_rank_pct": _get(target, "risk_rank_pct"),
                "adv20_amount": _get(target, "adv20_amount"),
                "estimated_price": estimated_price,
                "estimated_qty": estimated_qty,
                "target_value": target_value,
                "portfolio_id": _get(target, "portfolio_id") or config.portfolio_id,
                "entry_reason": _get(target, "entry_reason"),
                "hold_reason": _get(target, "hold_reason"),
                "exit_reason": _get(target, "exit_reason"),
                "sell_blocked_reason": _get(target, "sell_blocked_reason"),
                "signal_action": _get(target, "signal_action"),
                "target_exposure_scalar": _get(target, "target_exposure_scalar"),
                "status": "planned",
                "generated_at": now,
            }
        )
    planned = pd.DataFrame(rows)
    if not planned.empty:
        upsert_dataframe(con, "live_sim_planned_orders", planned, ["account_id", "decision_date", "code"])
    return planned


def _settle_due_orders(
    con: duckdb.DuckDBPyConnection,
    as_of_date: str,
    bars: pd.DataFrame,
    config: LiveSimConfig,
) -> pd.DataFrame:
    due = con.execute(
        """
        select *
        from live_sim_planned_orders
        where account_id = ? and execution_date = ?
          and not exists (
              select 1 from live_sim_executions e
              where e.account_id = live_sim_planned_orders.account_id
                and e.decision_date = live_sim_planned_orders.decision_date
                and e.code = live_sim_planned_orders.code
          )
        order by decision_date, code
        """,
        [config.account_id, as_of_date],
    ).fetchdf()
    if due.empty:
        return pd.DataFrame()
    holdings = _load_holdings(con, config.account_id)
    cash = _latest_cash(con, config)
    nav = _latest_nav(con, config)
    targets = due.rename(columns={"decision_date": "trade_date"})[
        [
            column
            for column in [
                "trade_date",
                "code",
                "target_weight",
                "trade_score_v2",
                "entry_reason",
                "exit_reason",
                "signal_action",
                "absolute_rank_pct",
                "risk_rank_pct",
                "sell_blocked_reason",
                "portfolio_id",
            ]
            if column in due.rename(columns={"decision_date": "trade_date"}).columns
        ]
    ].copy()
    targets["portfolio_id"] = config.portfolio_id
    if not holdings.empty:
        due_codes = set(due["code"].astype(str))
        hold_rows = []
        for row in holdings.itertuples(index=False):
            code = str(row.code)
            if code in due_codes:
                continue
            hold_rows.append(
                {
                    "trade_date": str(due["decision_date"].iloc[0]),
                    "code": code,
                    "target_weight": 0.0,
                    "trade_score_v2": getattr(row, "entry_trade_score", None),
                    "entry_reason": getattr(row, "entry_reason", None),
                    "portfolio_id": config.portfolio_id,
                    "signal_action": "hold",
                }
            )
        if hold_rows:
            targets = pd.concat([targets, pd.DataFrame(hold_rows)], ignore_index=True)
    orders = simulate_rebalance_orders(targets, bars, holdings.rename(columns={"qty": "position_qty"}), nav, config.execution, decision_date=str(due["decision_date"].iloc[0]))
    if orders.empty:
        return orders
    orders = _apply_cash_limit_and_holdings(con, orders, cash, config)
    records = orders.copy()
    records["account_id"] = config.account_id
    records["generated_at"] = _now()
    keep = [
        "account_id",
        "decision_date",
        "sim_date",
        "code",
        "side",
        "qty",
        "target_weight",
        "fill_px",
        "commission",
        "stamp_duty",
        "fees",
        "status",
        "reason",
        "entry_date",
        "entry_price",
        "exit_date",
        "holding_days",
        "entry_trade_score",
        "exit_trade_score",
        "entry_reason",
        "exit_reason",
        "sell_blocked_reason",
        "strategy_id",
        "realized_pnl",
        "generated_at",
    ]
    upsert_dataframe(con, "live_sim_executions", records[[col for col in keep if col in records.columns]], ["account_id", "decision_date", "sim_date", "code", "side"])
    return records


def _apply_cash_limit_and_holdings(
    con: duckdb.DuckDBPyConnection,
    orders: pd.DataFrame,
    cash: float,
    config: LiveSimConfig,
) -> pd.DataFrame:
    holdings = _load_holdings(con, config.account_id)
    qty_by_code = {str(row.code): float(row.qty) for row in holdings.itertuples(index=False)} if not holdings.empty else {}
    meta_by_code = _rows_by_code(holdings)
    out_rows = []
    for row in orders.sort_values("side", ascending=False).itertuples(index=False):
        data = row._asdict()
        if data.get("status") != "filled":
            out_rows.append(data)
            continue
        code = str(data["code"])
        qty = float(data["qty"])
        fill_px = float(data["fill_px"])
        commission = 0.0
        stamp_duty = 0.0
        if data.get("side") == "sell":
            qty = min(qty, qty_by_code.get(code, 0.0))
            value = qty * fill_px
            commission = value * config.execution.commission_bps / 10000.0
            stamp_duty = value * config.execution.stamp_duty_bps / 10000.0
            cash += value - commission - stamp_duty
            qty_by_code[code] = qty_by_code.get(code, 0.0) - qty
        else:
            gross_multiplier = 1.0 + config.execution.commission_bps / 10000.0
            affordable = int((cash / (fill_px * gross_multiplier)) // config.execution.a_share_lot_size) * config.execution.a_share_lot_size if not config.execution.allow_fractional_shares else cash / (fill_px * gross_multiplier)
            qty = min(qty, max(affordable, 0.0))
            if qty <= 0:
                data["status"] = "rejected"
                data["reason"] = "insufficient_cash"
                data["fill_px"] = None
            else:
                value = qty * fill_px
                commission = value * config.execution.commission_bps / 10000.0
                cash -= value + commission
                qty_by_code[code] = qty_by_code.get(code, 0.0) + qty
                meta_by_code[code] = pd.Series(
                    {
                        "code": code,
                        "entry_date": data.get("sim_date"),
                        "entry_price": fill_px,
                        "entry_trade_score": data.get("entry_trade_score"),
                        "entry_reason": data.get("entry_reason"),
                    }
                )
        data["qty"] = qty
        data["commission"] = commission
        data["stamp_duty"] = stamp_duty
        data["fees"] = commission + stamp_duty
        out_rows.append(data)
    _replace_holdings(con, config.account_id, qty_by_code, meta_by_code)
    return pd.DataFrame(out_rows)


def _record_nav(
    con: duckdb.DuckDBPyConnection,
    as_of_date: str,
    bars: pd.DataFrame,
    config: LiveSimConfig,
    holdings: pd.DataFrame,
) -> dict[str, float | str]:
    existing = con.execute("select * from live_sim_nav where account_id = ? and sim_date = ?", [config.account_id, as_of_date]).fetchdf()
    if not existing.empty:
        return existing.iloc[0].to_dict()
    cash = _latest_cash(con, config)
    market_value = _holding_market_value(holdings, bars, as_of_date)
    nav = cash + market_value
    total_return = nav / config.initial_cash - 1.0 if config.initial_cash else 0.0
    prev_nav = con.execute("select nav from live_sim_nav where account_id = ? order by sim_date", [config.account_id]).fetchdf()
    peak = max([config.initial_cash] + [float(value) for value in prev_nav["nav"].dropna().tolist()] + [nav])
    max_drawdown = nav / peak - 1.0 if peak else 0.0
    row = {
        "account_id": config.account_id,
        "sim_date": as_of_date,
        "nav": nav,
        "cash": cash,
        "holding_market_value": market_value,
        "total_return": total_return,
        "max_drawdown": max_drawdown,
    }
    upsert_dataframe(con, "live_sim_nav", pd.DataFrame([row]), ["account_id", "sim_date"])
    return row


def _sim_trading_dates(con: duckdb.DuckDBPyConnection, account_id: str, as_of_date: str) -> list[str]:
    dates = con.execute(
        """
        select sim_date
        from live_sim_nav
        where account_id = ?
          and sim_date <> 'INITIAL'
          and sim_date <= ?
        order by sim_date
        """,
        [account_id, as_of_date],
    ).fetchdf()
    values = {str(value)[:10] for value in dates["sim_date"].dropna().tolist()}
    values.add(as_of_date)
    return sorted(values)


def _refresh_holding_path_stats(
    con: duckdb.DuckDBPyConnection,
    account_id: str,
    holdings: pd.DataFrame,
    bars: pd.DataFrame,
    as_of_date: str,
) -> pd.DataFrame:
    if holdings.empty:
        con.execute("delete from live_sim_holding_path_stats where account_id = ?", [account_id])
        return holdings

    held_codes = holdings["code"].dropna().astype(str).tolist()
    codes_frame = pd.DataFrame({"code": held_codes})
    con.register("_live_sim_held_codes", codes_frame)
    try:
        con.execute(
            """
            delete from live_sim_holding_path_stats
            where account_id = ?
              and code not in (select code from _live_sim_held_codes)
            """,
            [account_id],
        )
    finally:
        con.unregister("_live_sim_held_codes")

    latest_bars = _latest_bars_by_code(bars, as_of_date)
    existing = _rows_by_code(
        con.execute(
            "select * from live_sim_holding_path_stats where account_id = ?",
            [account_id],
        ).fetchdf()
    )
    rows = []
    now = _now()
    for row in holdings.itertuples(index=False):
        code = str(row.code)
        entry_price = _float_or_none(getattr(row, "entry_price", None))
        bar = latest_bars.get(code)
        if entry_price is None or entry_price <= 0 or bar is None:
            continue
        close_px = _float_or_none(bar.get("close"))
        high_px = _float_or_none(bar.get("high"))
        low_px = _float_or_none(bar.get("low"))
        high_px = close_px if high_px is None else high_px
        low_px = close_px if low_px is None else low_px
        if close_px is None or close_px <= 0:
            continue
        current_close_ret = close_px / entry_price - 1.0
        high_ret = high_px / entry_price - 1.0 if high_px is not None else current_close_ret
        low_ret = low_px / entry_price - 1.0 if low_px is not None else current_close_ret
        old = existing.get(code)
        max_close_ret = max(_float_or_default(_get(old, "max_close_ret"), current_close_ret), current_close_ret)
        max_high_ret = max(_float_or_default(_get(old, "max_high_ret"), high_ret), high_ret)
        min_low_ret = min(_float_or_default(_get(old, "min_low_ret"), low_ret), low_ret)
        rows.append(
            {
                "account_id": account_id,
                "code": code,
                "max_close_ret": max_close_ret,
                "max_high_ret": max_high_ret,
                "current_close_ret": current_close_ret,
                "min_low_ret": min_low_ret,
                "updated_at": now,
            }
        )
    if rows:
        upsert_dataframe(con, "live_sim_holding_path_stats", pd.DataFrame(rows), ["account_id", "code"])
    return _load_holdings(con, account_id)


def _apply_profit_protection(
    targets: pd.DataFrame,
    holdings: pd.DataFrame,
    config: LiveSimConfig,
    as_of_date: str,
) -> pd.DataFrame:
    if not config.profit_protect_enabled or holdings.empty:
        return targets
    attrs = dict(targets.attrs)
    out = targets.copy()
    if out.empty:
        out = pd.DataFrame(columns=TARGET_COLUMNS)
    generated_at = _now()
    target_by_code = _target_index_by_code(out)
    protect_rows: list[dict[str, object]] = []
    for row in holdings.itertuples(index=False):
        code = str(row.code)
        holding_days = int(getattr(row, "holding_days", 0) or 0)
        if holding_days < config.profit_protect_min_days:
            continue
        max_gain = _float_or_none(getattr(row, "max_high_ret", None))
        if max_gain is None:
            max_gain = _float_or_none(getattr(row, "max_close_ret", None))
        current_ret = _float_or_none(getattr(row, "current_close_ret", None))
        if max_gain is None or current_ret is None:
            continue
        if max_gain < config.profit_protect_min_gain or current_ret > config.profit_protect_exit_below:
            continue
        idx = target_by_code.get(code)
        if idx is not None:
            signal = str(out.at[idx, "signal_action"]) if "signal_action" in out else ""
            reason = str(out.at[idx, "exit_reason"]) if "exit_reason" in out and pd.notna(out.at[idx, "exit_reason"]) else ""
            if signal == "sell" and reason not in {"not_candidate_after_target_days"}:
                continue
            out.at[idx, "signal_action"] = "sell"
            out.at[idx, "target_weight"] = 0.0
            out.at[idx, "entry_reason"] = "sell_signal"
            out.at[idx, "hold_reason"] = None
            out.at[idx, "exit_reason"] = "profit_protect_exit"
            out.at[idx, "sell_blocked_reason"] = None
            continue
        protect_rows.append(
            {
                "trade_date": as_of_date,
                "portfolio_id": config.portfolio_id,
                "code": code,
                "target_weight": 0.0,
                "rank_n": None,
                "trade_score": getattr(row, "latest_trade_score", getattr(row, "entry_trade_score", None)),
                "entry_reason": "sell_signal",
                "signal_action": "sell",
                "hold_reason": None,
                "exit_reason": "profit_protect_exit",
                "sell_blocked_reason": None,
                "entry_date": getattr(row, "entry_date", None),
                "entry_price": getattr(row, "entry_price", None),
                "shares": getattr(row, "shares", getattr(row, "qty", None)),
                "holding_days": holding_days,
                "entry_trade_score": getattr(row, "entry_trade_score", None),
                "latest_trade_score": getattr(row, "latest_trade_score", getattr(row, "entry_trade_score", None)),
                "generated_at": generated_at,
            }
        )
    if protect_rows:
        out = pd.concat([out, pd.DataFrame(protect_rows)], ignore_index=True)
    out = _ordered_target_columns(out)
    out.attrs.update(attrs)
    return out


def _market_exposure_scalar(bars: pd.DataFrame, as_of_date: str, config: LiveSimConfig) -> float:
    if config.market_zero_below is None and config.market_half_below is None:
        return 1.0
    if bars.empty or "trade_date" not in bars or "close" not in bars or "prev_close" not in bars:
        return 1.0
    frame = bars.copy()
    frame["_date"] = frame["trade_date"].astype(str).str[:10]
    prior_dates = sorted({date for date in frame["_date"].dropna().tolist() if date < as_of_date})
    if not prior_dates:
        return 1.0
    prev_date = prior_dates[-1]
    day = frame[frame["_date"] == prev_date].copy()
    close = pd.to_numeric(day["close"], errors="coerce")
    prev_close = pd.to_numeric(day["prev_close"], errors="coerce")
    valid = (close > 0) & (prev_close > 0)
    if "is_st" in day:
        valid &= ~day["is_st"].fillna(False).astype(bool)
    if "is_paused" in day:
        valid &= ~day["is_paused"].fillna(False).astype(bool)
    if "is_bse" in day:
        valid &= ~day["is_bse"].fillna(False).astype(bool)
    if "adv20_amount" in day:
        valid &= pd.to_numeric(day["adv20_amount"], errors="coerce").fillna(0.0) >= config.market_min_adv20_amount
    if not valid.any():
        return 1.0
    up_ratio = float((close[valid] > prev_close[valid]).mean())
    if config.market_zero_below is not None and up_ratio < config.market_zero_below:
        return 0.0
    if config.market_half_below is not None and up_ratio < config.market_half_below:
        return 0.5
    return 1.0


def _scale_target_exposure(targets: pd.DataFrame, exposure_scalar: float) -> pd.DataFrame:
    attrs = dict(targets.attrs)
    out = targets.copy()
    if out.empty:
        out.attrs.update(attrs)
        return out
    scalar = max(min(float(exposure_scalar), 1.0), 0.0)
    active_mask = pd.Series(True, index=out.index)
    if "signal_action" in out:
        active_mask &= out["signal_action"].astype(str) != "sell"
    out.loc[active_mask, "target_weight"] = pd.to_numeric(out.loc[active_mask, "target_weight"], errors="coerce").fillna(0.0) * scalar
    out.attrs.update(attrs)
    return out


def _latest_cash(con: duckdb.DuckDBPyConnection, config: LiveSimConfig) -> float:
    cashflow = con.execute(
        """
        select
            coalesce(sum(case when side = 'sell' and status = 'filled' then qty * fill_px else 0 end), 0) as sell_value,
            coalesce(sum(case when side = 'buy' and status = 'filled' then qty * fill_px else 0 end), 0) as buy_value,
            coalesce(sum(case when status = 'filled' then fees else 0 end), 0) as fees
        from live_sim_executions
        where account_id = ?
        """,
        [config.account_id],
    ).fetchone()
    if cashflow is None:
        return config.initial_cash
    return config.initial_cash + float(cashflow[0] or 0.0) - float(cashflow[1] or 0.0) - float(cashflow[2] or 0.0)


def _latest_nav(con: duckdb.DuckDBPyConnection, config: LiveSimConfig) -> float:
    latest = con.execute("select nav from live_sim_nav where account_id = ? order by sim_date desc limit 1", [config.account_id]).fetchone()
    return float(latest[0]) if latest else config.initial_cash


def _load_holdings(con: duckdb.DuckDBPyConnection, account_id: str) -> pd.DataFrame:
    return con.execute(
        """
        select h.*, s.max_close_ret, s.max_high_ret, s.current_close_ret, s.min_low_ret
        from live_sim_holdings h
        left join live_sim_holding_path_stats s
          on h.account_id = s.account_id and h.code = s.code
        where h.account_id = ? and h.qty > 0
        order by h.code
        """,
        [account_id],
    ).fetchdf()


def _replace_holdings(
    con: duckdb.DuckDBPyConnection,
    account_id: str,
    qty_by_code: dict[str, float],
    meta_by_code: dict[str, pd.Series],
) -> None:
    con.execute("delete from live_sim_holdings where account_id = ?", [account_id])
    rows = []
    now = _now()
    for code, qty in sorted(qty_by_code.items()):
        if qty <= 0:
            continue
        meta = meta_by_code.get(code, pd.Series({"code": code}))
        rows.append(
            {
                "account_id": account_id,
                "code": code,
                "qty": qty,
                "entry_date": _get(meta, "entry_date"),
                "entry_price": _get(meta, "entry_price"),
                "entry_trade_score": _get(meta, "entry_trade_score"),
                "entry_reason": _get(meta, "entry_reason"),
                "updated_at": now,
            }
        )
    if rows:
        upsert_dataframe(con, "live_sim_holdings", pd.DataFrame(rows), ["account_id", "code"])


def _holding_market_value(holdings: pd.DataFrame, bars: pd.DataFrame, as_of_date: str) -> float:
    if holdings.empty or bars.empty:
        return 0.0
    prices = (
        bars[bars["trade_date"].astype(str) <= as_of_date]
        .sort_values(["code", "trade_date"])
        .drop_duplicates("code", keep="last")
        .set_index("code")["close"]
        .to_dict()
    )
    return sum(float(row.qty) * float(prices.get(str(row.code), 0.0)) for row in holdings.itertuples(index=False))


def _annotate_holdings_for_report(holdings: pd.DataFrame, bars: pd.DataFrame, as_of_date: str) -> pd.DataFrame:
    if holdings.empty:
        return holdings
    out = holdings.copy()
    prices = _latest_close_prices(bars, as_of_date)
    out["current_price"] = out["code"].astype(str).map(prices).fillna(0.0)
    out["market_value"] = pd.to_numeric(out["qty"], errors="coerce").fillna(0.0) * out["current_price"]
    return out


def _latest_close_prices(bars: pd.DataFrame, as_of_date: str) -> dict[str, float]:
    if bars.empty:
        return {}
    frame = bars[bars["trade_date"].astype(str) <= as_of_date].sort_values(["code", "trade_date"])
    if frame.empty:
        return {}
    return {str(row.code): float(row.close) for row in frame.drop_duplicates("code", keep="last").itertuples(index=False)}


def _latest_bars_by_code(bars: pd.DataFrame, as_of_date: str) -> dict[str, pd.Series]:
    if bars.empty or "trade_date" not in bars or "code" not in bars:
        return {}
    frame = bars[bars["trade_date"].astype(str).str[:10] <= as_of_date].sort_values(["code", "trade_date"])
    if frame.empty:
        return {}
    return {
        str(row["code"]): row
        for _, row in frame.drop_duplicates("code", keep="last").iterrows()
    }


def _estimated_qty(
    side: str,
    target_value: float,
    estimated_price: float | None,
    holdings: pd.DataFrame,
    code: str,
    config: LiveSimConfig,
) -> float | None:
    if side == "sell":
        if holdings.empty:
            return 0.0
        row = holdings[holdings["code"].astype(str) == code]
        return float(row["qty"].iloc[0]) if not row.empty else 0.0
    if estimated_price is None or estimated_price <= 0:
        return None
    qty = target_value / estimated_price
    if not config.execution.allow_fractional_shares:
        qty = int(qty // config.execution.a_share_lot_size) * config.execution.a_share_lot_size
    return float(qty)


def _next_trading_day(as_of_date: str, bars: pd.DataFrame) -> str:
    if not bars.empty and "trade_date" in bars:
        future = sorted({str(value)[:10] for value in bars["trade_date"].dropna().tolist() if str(value)[:10] > as_of_date})
        if future:
            return future[0]
    day = datetime.fromisoformat(as_of_date).date() + timedelta(days=1)
    while day.weekday() >= 5:
        day += timedelta(days=1)
    return day.isoformat()


def _holdings_for_constructor(holdings: pd.DataFrame, as_of_date: str, trading_dates: list[str] | None = None) -> pd.DataFrame:
    if holdings.empty:
        return holdings
    out = holdings.rename(columns={"qty": "shares"}).copy()
    dates = trading_dates or [as_of_date]
    out["holding_days"] = out["entry_date"].map(lambda value: _trading_days_held(value, as_of_date, dates))
    out["calendar_days"] = out["entry_date"].map(lambda value: _calendar_days_held(value, as_of_date))
    out["latest_trade_score"] = out.get("entry_trade_score")
    return out


def _trading_days_held(entry_date: object, as_of_date: str, trading_dates: list[str]) -> int:
    start = pd.to_datetime(entry_date, errors="coerce")
    end = pd.to_datetime(as_of_date, errors="coerce")
    if pd.isna(start) or pd.isna(end) or end < start:
        return 0
    start_key = start.date().isoformat()
    end_key = end.date().isoformat()
    return sum(1 for date in trading_dates if start_key < str(date)[:10] <= end_key)


def _calendar_days_held(entry_date: object, as_of_date: str) -> int:
    start = pd.to_datetime(entry_date, errors="coerce")
    end = pd.to_datetime(as_of_date, errors="coerce")
    if pd.isna(start) or pd.isna(end) or end < start:
        return 0
    return int((end.date() - start.date()).days) + 1


def _rows_by_code(frame: pd.DataFrame) -> dict[str, pd.Series]:
    if frame.empty or "code" not in frame:
        return {}
    return {str(row["code"]): row for _, row in frame.iterrows()}


def _target_index_by_code(frame: pd.DataFrame) -> dict[str, object]:
    if frame.empty or "code" not in frame:
        return {}
    return {str(row["code"]): idx for idx, row in frame.iterrows()}


def _ordered_target_columns(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = [column for column in TARGET_COLUMNS if column in frame.columns]
    rest = [column for column in frame.columns if column not in ordered]
    return frame[ordered + rest].copy()


def _get(row: pd.Series | None, *columns: str) -> object | None:
    if row is None:
        return None
    for column in columns:
        if column in row and not pd.isna(row[column]):
            return row[column]
    return None


def _float_or_none(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _float_or_default(value: object, default: float) -> float:
    number = _float_or_none(value)
    return default if number is None else number


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return "无"
    keep = [column for column in columns if column in frame.columns]
    if not keep:
        return "无"
    rows = ["| " + " | ".join(keep) + " |", "| " + " | ".join(["---"] * len(keep)) + " |"]
    for _, row in frame[keep].iterrows():
        rows.append("| " + " | ".join(_format_cell(row[column]) for column in keep) + " |")
    return "\n".join(rows)


def _format_cell(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:,.4f}"
    return str(value)


def _money(value: object) -> str:
    return f"{float(value or 0.0):,.2f}"


def _pct(value: object) -> str:
    return f"{float(value or 0.0) * 100:.2f}%"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_manifest(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _snapshot_live_model_artifacts(
    *,
    snapshot_dir: Path,
    manifest_path: Path,
    alpha_artifact: Path,
    risk_artifact: Path,
    feature_schema: Path,
) -> dict[str, Path]:
    for path in [manifest_path, alpha_artifact, risk_artifact, feature_schema]:
        if not path.exists():
            raise FileNotFoundError(path)

    snapshot_dir.mkdir(parents=True, exist_ok=True)
    alpha_target = _copy_live_artifact_with_params(alpha_artifact, snapshot_dir)
    risk_target = _copy_live_artifact_with_params(risk_artifact, snapshot_dir)
    schema_target = snapshot_dir / "feature_schema.json"
    manifest_target = snapshot_dir / "manifest.json"
    shutil.copy2(feature_schema, schema_target)
    shutil.copy2(manifest_path, manifest_target)
    return {
        "alpha_artifact_uri": alpha_target,
        "risk_artifact_uri": risk_target,
        "feature_schema_uri": schema_target,
        "source_manifest_path": manifest_target,
    }


def _copy_live_artifact_with_params(source: Path, target_dir: Path) -> Path:
    target = target_dir / source.name
    shutil.copy2(source, target)
    params = source.with_suffix(".params.json")
    if params.exists():
        shutil.copy2(params, target_dir / params.name)
    return target


def _upsert_live_model_bundle(con: duckdb.DuckDBPyConnection, bundle: dict[str, object]) -> None:
    con.execute(
        """
        update live_model_bundle
        set is_active = false,
            deactivated_at = ?
        where strategy_id = ?
          and is_active = true
          and bundle_id <> ?
        """,
        [_now(), bundle["strategy_id"], bundle["bundle_id"]],
    )
    columns = [
        "bundle_id",
        "strategy_id",
        "score_version",
        "source_run_id",
        "source_fold_id",
        "source_manifest_path",
        "source_manifest_hash",
        "alpha_model_id",
        "risk_model_id",
        "alpha_artifact_uri",
        "risk_artifact_uri",
        "feature_schema_uri",
        "feature_set_id",
        "label_base",
        "horizon_d",
        "train_window_mode",
        "source_train_window_mode",
        "alpha_rounds",
        "risk_rounds",
        "activated_at",
        "deactivated_at",
        "is_active",
        "notes",
    ]
    placeholders = ", ".join(["?"] * len(columns))
    updates = ", ".join(f"{column} = excluded.{column}" for column in columns[1:])
    con.execute(
        f"""
        insert into live_model_bundle ({", ".join(columns)})
        values ({placeholders})
        on conflict (bundle_id) do update set {updates}
        """,
        [bundle.get(column) for column in columns],
    )


def _load_live_model_bundle_by_id(con: duckdb.DuckDBPyConnection, bundle_id: str) -> dict[str, object]:
    row = con.execute(
        "select * from live_model_bundle where bundle_id = ?",
        [bundle_id],
    ).fetchdf()
    if row.empty:
        raise RuntimeError(f"live model bundle not found: {bundle_id}")
    return row.iloc[0].to_dict()
