from __future__ import annotations

import argparse
from pathlib import Path
import sys

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import research_risk_controls as rc


DEFAULT_RUN_ID = "wf_v2_ret5_fund_fixed_a160_r120_20260621"
DEFAULT_SCORE_VERSION = "v2_alpha_ret5d_fund_fixed_a160_r120_20260621"
DEFAULT_VARIANT = "mkt_tier_profit_protect"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ml-db", default="outputs/ml/ml_ret5_alpha_risk_20260619.duckdb")
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--score-version", default=DEFAULT_SCORE_VERSION)
    parser.add_argument("--base-portfolio-id", default="continuous_wf_replay")
    parser.add_argument("--variant", default=DEFAULT_VARIANT)
    parser.add_argument("--start-year", type=int, default=2020)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--initial-cash", type=float, default=1_000_000.0)
    parser.add_argument("--slippage-bps", type=float, default=10.0)
    parser.add_argument("--commission-bps", type=float, default=3.0)
    parser.add_argument("--stamp-duty-bps", type=float, default=5.0)
    parser.add_argument(
        "--out-dir",
        default="outputs/ml/reports/profit_protect_continuous_wf_2020_2025_20260622",
    )
    return parser


def main() -> None:
    main_with_args(None)


def main_with_args(argv: list[str] | None) -> None:
    args = build_arg_parser().parse_args(argv)
    variant = _variant_by_name(args.variant)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rc._set_run_context(args.run_id, args.score_version, args.base_portfolio_id)
    execution = rc.ExecutionConfig(
        slippage_bps=float(args.slippage_bps),
        commission_bps=float(args.commission_bps),
        stamp_duty_bps=float(args.stamp_duty_bps),
        allow_fractional_shares=False,
    )
    con = duckdb.connect(args.ml_db, read_only=True)
    try:
        scored = _load_scored_candidates(con, args)
        bars = _load_bars(con, args)
        market_state = _load_market_state(con, args)
    finally:
        con.close()

    result = run_continuous_backtest(scored, bars, market_state, variant, args, execution)
    summary = _continuous_summary(result, args)
    yearly = _continuous_yearly(result, args)
    exit_summary = _exit_summary(result.orders)

    result.nav.to_csv(out_dir / "continuous_nav.csv", index=False)
    result.orders.to_csv(out_dir / "continuous_orders.csv", index=False)
    result.positions.to_csv(out_dir / "continuous_positions.csv", index=False)
    diagnostics = result.portfolio_diagnostics if result.portfolio_diagnostics is not None else pd.DataFrame()
    diagnostics.to_csv(out_dir / "continuous_diagnostics.csv", index=False)
    summary.to_csv(out_dir / "continuous_summary.csv", index=False)
    yearly.to_csv(out_dir / "continuous_yearly.csv", index=False)
    exit_summary.to_csv(out_dir / "exit_reason_summary.csv", index=False)

    print(summary.to_string(index=False))
    print(yearly.to_string(index=False))
    print(f"wrote {out_dir}")


def run_continuous_backtest(
    scored_candidates: pd.DataFrame,
    bars: pd.DataFrame,
    market_state: dict[str, dict[str, float]],
    variant: rc.Variant,
    args: argparse.Namespace,
    execution: rc.ExecutionConfig,
) -> rc.BacktestResult:
    constraints = rc._constraints_for_variant(variant)
    config = rc.BacktestConfig(
        initial_cash=float(args.initial_cash),
        portfolio_id=f"{args.base_portfolio_id}_{variant.name}_{args.start_year}_{args.end_year}",
        execution=execution,
        decision_dates=sorted(scored_candidates["trade_date"].dropna().unique()),
    )
    scored = scored_candidates.copy()
    candidates_by_date = rc._group_by_trade_date(scored)
    bars_sorted = bars.sort_values(["code", "trade_date"]).copy()
    bars_sorted.attrs["_vpa_sorted_by_code_date"] = True
    bar_index = rc.BarDataIndex.from_bars(bars_sorted)
    trading_dates = sorted(bars_sorted["trade_date"].dropna().unique())
    decision_dates = rc._date_key_set(scored["trade_date"].dropna().unique())
    daily_bars = rc._daily_bar_lookup(bars_sorted)

    cash = config.initial_cash
    positions: dict[str, float] = {}
    position_meta: dict[str, dict[str, object]] = {}
    pending_orders: list[dict[str, object]] = []
    all_orders: list[dict[str, object]] = []
    nav_rows: list[dict[str, object]] = []
    position_rows: list[dict[str, object]] = []
    diagnostic_frames: list[pd.DataFrame] = []
    next_order_seq = 1
    peak_nav = config.initial_cash

    for date in trading_dates:
        year = int(str(date)[:4])
        todays_orders = [order for order in pending_orders if order.get("sim_date") == date]
        pending_orders = [order for order in pending_orders if order.get("sim_date") != date]
        turnover_value = 0.0
        for order in sorted(todays_orders, key=rc._execution_priority):
            if order.get("status") != "filled":
                continue
            qty = float(order["qty"])
            fill_px = float(order["fill_px"])
            if order.get("side") == "buy":
                qty = rc._cash_limited_buy_qty_with_fees(qty, fill_px, cash, config.execution)
                if qty <= 1e-12:
                    order["qty"] = 0.0
                    order["fill_px"] = None
                    order["status"] = "rejected"
                    order["reason"] = "insufficient_cash"
                    continue
                order["qty"] = qty
            value = qty * fill_px
            commission, stamp_duty = rc._execution_fees(value, str(order.get("side")), config.execution)
            order["commission"] = commission
            order["stamp_duty"] = stamp_duty
            order["fees"] = commission + stamp_duty
            order["slippage_bps"] = config.execution.slippage_bps
            turnover_value += value
            code = str(order["code"])
            if order.get("side") == "sell":
                cash += value - commission - stamp_duty
                next_qty = positions.get(code, 0.0) - qty
            else:
                cash -= value + commission
                next_qty = positions.get(code, 0.0) + qty
                new_meta = {
                    "entry_date": order.get("sim_date"),
                    "entry_price": fill_px,
                    "entry_trade_score": order.get("entry_trade_score"),
                    "entry_abs_rank_pct": order.get("entry_abs_rank_pct"),
                    "entry_risk_rank_pct": order.get("entry_risk_rank_pct"),
                    "entry_reason": order.get("entry_reason"),
                    "holding_days": 0,
                    "last_holding_update": order.get("sim_date"),
                    "max_close_ret": 0.0,
                    "max_high_ret": 0.0,
                    "current_close_ret": 0.0,
                    "min_low_ret": 0.0,
                }
                position_meta.setdefault(code, new_meta)
            if abs(next_qty) <= 1e-9:
                positions.pop(code, None)
                position_meta.pop(code, None)
            else:
                positions[code] = next_qty

        rc._advance_trading_holding_days(position_meta, str(date))
        rc._update_position_path_stats(positions, position_meta, daily_bars, date)
        gross = rc._position_value(positions, bar_index, date)
        nav = cash + gross
        peak_nav = max(peak_nav, nav)
        account_drawdown = nav / peak_nav - 1.0 if peak_nav else 0.0
        exposure_scalar = rc._exposure_scalar(variant, str(date), market_state, account_drawdown)
        nav_rows.append(
            {
                "run_id": args.run_id,
                "sim_date": date,
                "nav": nav,
                "cash": cash,
                "gross_exposure": gross / nav if nav else 0.0,
                "target_exposure_scalar": exposure_scalar,
                "account_drawdown": account_drawdown,
                "turnover": turnover_value / nav if nav else 0.0,
                "period": year,
                "variant": variant.name,
            }
        )
        for code, qty in positions.items():
            px = bar_index.close_at_or_before(code, date)
            meta = position_meta.get(code, {})
            position_rows.append(
                {
                    "run_id": args.run_id,
                    "sim_date": date,
                    "code": code,
                    "position_qty": qty,
                    "market_value": qty * px,
                    "weight": (qty * px / nav) if nav else 0.0,
                    "entry_date": meta.get("entry_date"),
                    "entry_price": meta.get("entry_price"),
                    "holding_days": meta.get("holding_days", 0),
                    "entry_trade_score": meta.get("entry_trade_score"),
                    "entry_abs_rank_pct": meta.get("entry_abs_rank_pct"),
                    "entry_risk_rank_pct": meta.get("entry_risk_rank_pct"),
                    "entry_reason": meta.get("entry_reason"),
                    "period": year,
                    "variant": variant.name,
                }
            )

        date_key = rc._date_key(date)
        if date_key in decision_dates:
            pending_codes = {str(order["code"]) for order in pending_orders}
            day_candidates = candidates_by_date.get(date_key)
            if day_candidates is None:
                day_candidates = pd.DataFrame(columns=scored.columns)
            else:
                day_candidates = day_candidates.copy()
            holdings = rc._fixed_holdings_frame(positions, position_meta)
            targets = rc.construct_portfolio_targets_v2(
                day_candidates,
                constraints,
                config.portfolio_id,
                current_holdings=holdings,
                run_id=args.run_id,
                fold_id=f"wf_{year}",
                score_version=variant.name,
            )
            targets = rc._enrich_targets_from_candidates(targets, day_candidates)
            targets = rc._apply_profit_protection(targets, holdings, date, variant, config.portfolio_id)
            targets = rc._apply_selective_not_candidate_grace(targets, holdings, variant)
            targets = rc._apply_stop_loss(targets, holdings, bar_index, date, variant, config.portfolio_id)
            current_weights = rc._current_position_weights(positions, bar_index, date, nav)
            targets = rc._allocate_variant_weights(targets, variant, constraints, exposure_scalar, current_weights)
            diagnostics = rc.get_portfolio_diagnostics(targets)
            if not diagnostics.empty:
                diagnostics = diagnostics.copy()
                diagnostics["period"] = year
                diagnostics["variant"] = variant.name
                diagnostics["target_exposure_scalar"] = exposure_scalar
                diagnostic_frames.append(diagnostics)
            orders = rc.simulate_rebalance_orders(
                targets,
                bars_sorted,
                rc._positions_frame(positions, position_meta),
                nav,
                config.execution,
                decision_date=str(date),
                bar_index=bar_index,
            )
            if pending_codes and not orders.empty:
                orders = orders[~orders["code"].astype(str).isin(pending_codes)].reset_index(drop=True)
            if not orders.empty:
                records = orders.to_dict("records")
                for record in records:
                    record["order_seq"] = next_order_seq
                    record["period"] = year
                    record["variant"] = variant.name
                    record["run_id"] = args.run_id
                    record["fold_id"] = f"wf_{year}"
                    record["strategy_id"] = variant.name
                    record["score_version"] = variant.name
                    next_order_seq += 1
                all_orders.extend(records)
                pending_orders.extend(records)

    return rc.BacktestResult(
        rc._with_realized_returns(pd.DataFrame(all_orders)),
        pd.DataFrame(position_rows),
        pd.DataFrame(nav_rows),
        pd.concat(diagnostic_frames, ignore_index=True) if diagnostic_frames else pd.DataFrame(),
    )


def _variant_by_name(name: str) -> rc.Variant:
    for variant in rc._variants():
        if variant.name == name:
            return variant
    raise ValueError(f"unknown variant: {name}")


def _load_scored_candidates(con: duckdb.DuckDBPyConnection, args: argparse.Namespace) -> pd.DataFrame:
    frames = []
    for year in range(args.start_year, args.end_year + 1):
        fold_id = f"wf_{year}"
        frame = con.execute(
            """
            select
                p.*,
                t.industry_code,
                t.industry_name,
                t.is_st,
                t.is_paused,
                t.is_bse,
                t.adv20_amount,
                t.can_buy_next_open,
                t.can_sell_next_open,
                t.next_open,
                t.next_limit_up,
                t.next_limit_down,
                t.next_is_paused,
                t.limit_up_pct,
                t.limit_down_pct,
                t.limit_band
            from ml_predictions_daily p
            left join ml_tradeability_daily t
              on p.trade_date = t.trade_date and p.code = t.code
            where p.run_id = ?
              and p.fold_id = ?
              and p.score_version = ?
              and p.trade_date between ? and ?
            order by p.trade_date, p.code
            """,
            [args.run_id, fold_id, args.score_version, f"{year}-01-01", f"{year}-12-31"],
        ).fetchdf()
        if frame.empty:
            raise RuntimeError(f"no candidates loaded for {fold_id}")
        scored = rc.archived_adv_score(frame, args.score_version)
        scored["run_id"] = args.run_id
        scored["fold_id"] = fold_id
        frames.append(scored)
    return pd.concat(frames, ignore_index=True)


def _load_bars(con: duckdb.DuckDBPyConnection, args: argparse.Namespace) -> pd.DataFrame:
    frame = con.execute(
        """
        select
            trade_date,
            code,
            open,
            high,
            low,
            close,
            limit_up,
            limit_down,
            is_paused
        from ml_tradeability_daily
        where trade_date between ? and ?
        order by code, trade_date
        """,
        [f"{args.start_year}-01-01", f"{args.end_year}-12-31"],
    ).fetchdf()
    if frame.empty:
        raise RuntimeError("no bars loaded")
    return frame


def _load_market_state(con: duckdb.DuckDBPyConnection, args: argparse.Namespace) -> dict[str, dict[str, float]]:
    return rc._load_market_state(con, int(args.start_year), int(args.end_year))


def _continuous_summary(result: rc.BacktestResult, args: argparse.Namespace) -> pd.DataFrame:
    nav = result.nav
    total_return = float(nav["nav"].iloc[-1] / args.initial_cash - 1.0) if not nav.empty else 0.0
    annual_return = _annualized_by_dates(nav)
    drawdown = rc._max_drawdown(nav["nav"]) if not nav.empty else 0.0
    return pd.DataFrame(
        [
            {
                "variant": args.variant,
                "start_year": args.start_year,
                "end_year": args.end_year,
                "total_return": total_return,
                "annual_return": annual_return,
                "max_drawdown": drawdown,
                "avg_exposure": _avg_exposure(nav),
                **rc._trade_metrics(result.orders),
            }
        ]
    )


def _continuous_yearly(result: rc.BacktestResult, args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    nav = result.nav
    orders = result.orders
    prior_nav = float(args.initial_cash)
    for year in range(args.start_year, args.end_year + 1):
        year_nav = nav[nav["period"] == year].copy() if not nav.empty else pd.DataFrame()
        year_orders = orders[orders["period"] == year].copy() if not orders.empty and "period" in orders else pd.DataFrame()
        if year_nav.empty:
            continue
        ending_nav = float(year_nav["nav"].iloc[-1])
        rows.append(
            {
                "variant": args.variant,
                "year": year,
                "start_nav_basis": prior_nav,
                "ending_nav": ending_nav,
                "total_return": ending_nav / prior_nav - 1.0 if prior_nav else 0.0,
                "max_drawdown": rc._max_drawdown(year_nav["nav"]),
                "avg_exposure": _avg_exposure(year_nav),
                **rc._trade_metrics(year_orders),
            }
        )
        prior_nav = ending_nav
    return pd.DataFrame(rows)


def _exit_summary(orders: pd.DataFrame) -> pd.DataFrame:
    if orders.empty or "exit_reason" not in orders:
        return pd.DataFrame()
    sells = orders[(orders["side"].astype(str) == "sell") & (orders["status"].astype(str) == "filled")].copy()
    if sells.empty:
        return pd.DataFrame()
    sells["exit_reason"] = sells["exit_reason"].fillna("unknown")
    sells["realized_ret_num"] = pd.to_numeric(sells["realized_ret"], errors="coerce")
    return (
        sells.groupby("exit_reason", dropna=False)
        .agg(
            trade_count=("code", "count"),
            win_rate=("realized_ret_num", lambda value: float((value > 0).mean()) if len(value) else 0.0),
            avg_ret=("realized_ret_num", "mean"),
            total_realized_pnl=("realized_pnl", "sum"),
        )
        .reset_index()
        .sort_values("trade_count", ascending=False)
    )


def _annualized_by_dates(nav: pd.DataFrame) -> float:
    if nav.empty or len(nav) < 2:
        return 0.0
    ordered = nav.sort_values("sim_date")
    start = float(ordered["nav"].iloc[0])
    end = float(ordered["nav"].iloc[-1])
    dates = pd.to_datetime(ordered["sim_date"], errors="coerce").dropna()
    if start <= 0 or end <= 0 or len(dates) < 2:
        return 0.0
    years = (dates.iloc[-1] - dates.iloc[0]).days / 365.25
    return (end / start) ** (1.0 / years) - 1.0 if years > 0 else 0.0


def _avg_exposure(nav: pd.DataFrame) -> float:
    if nav.empty or "gross_exposure" not in nav:
        return 0.0
    return float(pd.to_numeric(nav["gross_exposure"], errors="coerce").fillna(0.0).mean())


if __name__ == "__main__":
    main()
