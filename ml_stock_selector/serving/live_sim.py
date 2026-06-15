from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import duckdb
import pandas as pd

from ml_stock_selector.backtest.execution import ExecutionConfig, simulate_rebalance_orders
from ml_stock_selector.portfolio.allocator import allocate_weights
from ml_stock_selector.portfolio.constraints import PortfolioConstraints
from ml_stock_selector.portfolio.constructor import construct_portfolio_targets_v2
from ml_stock_selector.portfolio.holding_policy import HoldingPolicy
from ml_stock_selector.storage import upsert_dataframe

SCORE_VERSION = "preferred_adv10m_fulladv015_top12"


@dataclass(frozen=True)
class LiveSimConfig:
    account_id: str = "preferred_adv10m_paper"
    initial_cash: float = 300_000.0
    portfolio_id: str = "preferred_adv10m_fulladv015_top12"
    target_positions: int = 12
    report_dir: Path = Path("outputs/ml/live_sim/reports")
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
        "score_version": SCORE_VERSION,
        "account_id": config.account_id,
        "portfolio_id": config.portfolio_id,
        "initial_cash": config.initial_cash,
        "target_positions": config.target_positions,
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
    return con


def archived_adv_score(predictions: pd.DataFrame) -> pd.DataFrame:
    out = predictions.copy()
    if out.empty:
        out["full_prediction_pool_adv_pct"] = pd.Series(dtype=float)
        out["trade_score_v2"] = pd.Series(dtype=float)
        out["alpha_rank_pct"] = pd.Series(dtype=float)
        out["active_rank_pct"] = pd.Series(dtype=float)
        out["core_score"] = pd.Series(dtype=float)
        out["trade_score"] = pd.Series(dtype=float)
        out["score_version"] = SCORE_VERSION
        return out
    out["absolute_rank_pct"] = pd.to_numeric(out["absolute_rank_pct"], errors="coerce").fillna(0.0)
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
    out["score_version"] = SCORE_VERSION
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
    nav = _record_nav(con, as_of_date, bars, config, holdings)
    execution_date = _next_trading_day(as_of_date, bars)
    targets = _build_targets(predictions, holdings, config, as_of_date, _sim_trading_dates(con, config.account_id, as_of_date))
    planned = _plan_orders(con, as_of_date, execution_date, targets, holdings, bars, float(nav["nav"]), config)
    holdings_report = _annotate_holdings_for_report(holdings, bars, as_of_date)
    result = LiveSimDayResult(config.account_id, as_of_date, execution_date, planned, executions, holdings_report, nav)
    report_path = config.report_dir / f"live_sim_summary_{as_of_date}.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(generate_markdown_report(result), encoding="utf-8")
    return LiveSimDayResult(config.account_id, as_of_date, execution_date, planned, executions, holdings, nav, report_path)


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
        _markdown_table(result.executions, ["code", "side", "qty", "fill_px", "fees", "status", "reason"]),
        "",
        "## 当前持仓",
        _markdown_table(result.holdings, ["code", "qty", "current_price", "market_value", "entry_price", "entry_date", "entry_trade_score"]),
        "",
        "## 下一交易日计划",
        f"- 执行日期: {result.execution_date}",
        _markdown_table(result.planned_orders, ["code", "side", "estimated_price", "estimated_qty", "target_value", "target_weight", "trade_score_v2", "adv20_amount", "status"]),
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
) -> pd.DataFrame:
    scored = archived_adv_score(predictions)
    # Signal generation must not depend on next-open tradeability columns, which are future
    # information at T-day close. Settlement handles limits and pauses with actual T+1 bars.
    scored = scored.drop(columns=[col for col in ["can_buy_next_open", "can_sell_next_open"] if col in scored], errors="ignore")
    targets = construct_portfolio_targets_v2(
        scored,
        config.constraints,
        config.portfolio_id,
        current_holdings=_holdings_for_constructor(holdings, as_of_date, trading_dates),
        score_version=SCORE_VERSION,
    )
    if targets.empty:
        return targets
    weighted = allocate_weights(targets, 1.0 / config.target_positions, 1.0 / config.target_positions, allow_cash=True)
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
                "entry_reason": _get(target, "entry_reason"),
                "signal_action": _get(target, "signal_action"),
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
        ["trade_date", "code", "target_weight", "trade_score_v2", "entry_reason"]
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
    keep = ["account_id", "decision_date", "sim_date", "code", "side", "qty", "target_weight", "fill_px", "commission", "stamp_duty", "fees", "status", "reason", "realized_pnl", "generated_at"]
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
    return con.execute("select * from live_sim_holdings where account_id = ? and qty > 0 order by code", [account_id]).fetchdf()


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


def _get(row: pd.Series | None, *columns: str) -> object | None:
    if row is None:
        return None
    for column in columns:
        if column in row and not pd.isna(row[column]):
            return row[column]
    return None


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
