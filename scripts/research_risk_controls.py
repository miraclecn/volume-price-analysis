from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path
import sys
from typing import Iterable

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml_stock_selector.backtest.engine import (
    BacktestConfig,
    BacktestResult,
    _date_key,
    _date_key_set,
    _execution_priority,
    _advance_trading_holding_days,
    _fixed_holdings_frame,
    _group_by_trade_date,
    _holdings_frame,
    _position_value,
    _positions_frame,
)
from ml_stock_selector.backtest.execution import BarDataIndex, ExecutionConfig, simulate_rebalance_orders
from ml_stock_selector.portfolio.constraints import PortfolioConstraints
from ml_stock_selector.portfolio.constructor import (
    TARGET_COLUMNS,
    construct_portfolio_targets_v2,
    get_portfolio_diagnostics,
)
from ml_stock_selector.portfolio.holding_policy import HoldingPolicy
from ml_stock_selector.serving.live_sim import archived_adv_score


RUN_ID = "wf_v2_fix12_full131_oldlabels_20260618"
SCORE_VERSION = "v2_three_model_full131_oldlabels_20260618"
BASE_PORTFOLIO_ID = "risk_control_replay"
PREDICTION_FOLD_ID: str | None = None


@dataclass(frozen=True)
class Variant:
    name: str
    stop_loss_pct: float | None = None
    stop_min_days: int = 0
    stop_max_days: int | None = None
    market_rule: str = "none"
    market_zero_below: float | None = None
    market_half_below: float | None = None
    account_dd_half_at: float | None = None
    weight_mode: str = "equal"
    exposure_mode: str = "portfolio"
    target_hold_days: int | None = None
    max_hold_days: int | None = None
    force_exit_after_max_hold_days: bool | None = None
    selective_not_candidate_grace: bool = False
    grace_max_hold_days: int = 8
    grace_min_trade_score: float = 0.55
    grace_max_risk_rank_pct: float = 0.55
    grace_min_low_adv_score: float = 0.80
    grace_min_current_ret: float = -0.08
    grace_max_current_ret: float = 0.0
    grace_max_prior_gain: float = 0.03
    profit_protect: bool = False
    profit_protect_min_days: int = 3
    profit_protect_min_gain: float = 0.03
    profit_protect_exit_below: float = 0.005


@dataclass(frozen=True)
class ResearchResult:
    variant: Variant
    year: int
    orders: pd.DataFrame
    nav: pd.DataFrame
    diagnostics: pd.DataFrame


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ml-db", default="outputs/ml/ml_full131_oldlabels_20260618.duckdb")
    parser.add_argument(
        "--out-dir",
        default="outputs/ml/reports/full131_oldlabels_risk_control_replay_2020_2025",
    )
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--score-version", default=SCORE_VERSION)
    parser.add_argument("--base-portfolio-id", default=BASE_PORTFOLIO_ID)
    parser.add_argument("--prediction-fold-id", help="Override prediction fold id, e.g. wf_2026_ytd.")
    parser.add_argument("--years", nargs="+", type=int, default=[2020, 2021, 2022, 2023, 2024, 2025])
    parser.add_argument("--variants", nargs="*", help="Optional subset of variant names to run.")
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--commission-bps", type=float, default=3.0)
    parser.add_argument("--stamp-duty-bps", type=float, default=5.0)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    _set_run_context(args.run_id, args.score_version, args.base_portfolio_id)
    _set_prediction_fold_id(args.prediction_fold_id)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    variants = _variants()
    if args.variants:
        wanted = set(args.variants)
        variants = [variant for variant in variants if variant.name in wanted]
        missing = wanted - {variant.name for variant in variants}
        if missing:
            raise ValueError(f"unknown variants: {sorted(missing)}")
    execution = ExecutionConfig(
        slippage_bps=float(args.slippage_bps),
        commission_bps=float(args.commission_bps),
        stamp_duty_bps=float(args.stamp_duty_bps),
    )

    con = duckdb.connect(args.ml_db, read_only=True)
    try:
        market_state = _load_market_state(con, min(args.years), max(args.years))
        all_results: list[ResearchResult] = []
        for year in args.years:
            scored = _load_scored_candidates(con, year)
            bars = _load_bars(con, year)
            for variant in variants:
                print(f"running year={year} variant={variant.name}", flush=True)
                result = run_research_backtest(scored, bars, market_state, variant, year, execution)
                all_results.append(result)
    finally:
        con.close()

    summary = _summarize_results(all_results)
    yearly = _summarize_years(all_results)
    orders = _concat_result_frames(all_results, "orders")
    nav = _concat_result_frames(all_results, "nav")
    diagnostics = _concat_result_frames(all_results, "diagnostics")

    summary.to_csv(out_dir / "risk_control_summary.csv", index=False)
    yearly.to_csv(out_dir / "risk_control_yearly.csv", index=False)
    orders.to_csv(out_dir / "risk_control_orders.csv", index=False)
    nav.to_csv(out_dir / "risk_control_nav.csv", index=False)
    diagnostics.to_csv(out_dir / "risk_control_diagnostics.csv", index=False)
    print(summary.sort_values(["loss_to_win", "annual_return"], ascending=[True, False]).to_string(index=False))
    print(f"wrote {out_dir}")


def _set_run_context(run_id: str, score_version: str, base_portfolio_id: str) -> None:
    global RUN_ID, SCORE_VERSION, BASE_PORTFOLIO_ID
    RUN_ID = str(run_id)
    SCORE_VERSION = str(score_version)
    BASE_PORTFOLIO_ID = str(base_portfolio_id)


def _set_prediction_fold_id(fold_id: str | None) -> None:
    global PREDICTION_FOLD_ID
    PREDICTION_FOLD_ID = str(fold_id) if fold_id else None


def _variants() -> list[Variant]:
    return [
        Variant("base"),
        Variant("stop005_d0", stop_loss_pct=-0.005, stop_min_days=0),
        Variant("stop010_d0", stop_loss_pct=-0.010, stop_min_days=0),
        Variant("stop015_d0", stop_loss_pct=-0.015, stop_min_days=0),
        Variant("stop020_d0", stop_loss_pct=-0.020, stop_min_days=0),
        Variant("stop020_d1", stop_loss_pct=-0.020, stop_min_days=1),
        Variant("stop030_d0", stop_loss_pct=-0.030, stop_min_days=0),
        Variant("stop050_d0", stop_loss_pct=-0.050, stop_min_days=0),
        Variant("stop050_h5", stop_loss_pct=-0.050, stop_min_days=0, stop_max_days=5),
        Variant("stop080_d0", stop_loss_pct=-0.080, stop_min_days=0),
        Variant("stop100_d0", stop_loss_pct=-0.100, stop_min_days=0),
        Variant("target10_max10", target_hold_days=10, max_hold_days=10),
        Variant("no_time_exit", force_exit_after_max_hold_days=False),
        Variant("target10_no_time_exit", target_hold_days=10, force_exit_after_max_hold_days=False),
        Variant("stop050_target10_max10", stop_loss_pct=-0.050, target_hold_days=10, max_hold_days=10),
        Variant("stop050_no_time_exit", stop_loss_pct=-0.050, force_exit_after_max_hold_days=False),
        Variant(
            "stop050_target10_no_time_exit",
            stop_loss_pct=-0.050,
            target_hold_days=10,
            force_exit_after_max_hold_days=False,
        ),
        Variant("mkt_half_prev_up45", market_rule="half_prev_up45"),
        Variant("mkt_tier_prev_up35_45", market_rule="tier_prev_up35_45"),
        Variant("mkt_tier_prev_up325_425", market_zero_below=0.325, market_half_below=0.425),
        Variant("mkt_tier_prev_up375_475", market_zero_below=0.375, market_half_below=0.475),
        Variant(
            "mkt_tier_profit_protect",
            market_zero_below=0.375,
            market_half_below=0.475,
            profit_protect=True,
        ),
        Variant(
            "mkt_tier_grace_lowadv_h8",
            market_zero_below=0.375,
            market_half_below=0.475,
            selective_not_candidate_grace=True,
            grace_max_hold_days=8,
        ),
        Variant(
            "mkt_tier_grace_lowadv_h10",
            market_zero_below=0.375,
            market_half_below=0.475,
            selective_not_candidate_grace=True,
            grace_max_hold_days=10,
        ),
        Variant(
            "mkt_tier_combo_h8",
            market_zero_below=0.375,
            market_half_below=0.475,
            selective_not_candidate_grace=True,
            grace_max_hold_days=8,
            profit_protect=True,
        ),
        Variant(
            "mkt_tier_combo_h10",
            market_zero_below=0.375,
            market_half_below=0.475,
            selective_not_candidate_grace=True,
            grace_max_hold_days=10,
            profit_protect=True,
        ),
        Variant(
            "mkt_tier_prev_up325_425_entry_throttle",
            market_zero_below=0.325,
            market_half_below=0.425,
            exposure_mode="entry_throttle",
        ),
        Variant(
            "mkt_tier_prev_up375_475_entry_throttle",
            market_zero_below=0.375,
            market_half_below=0.475,
            exposure_mode="entry_throttle",
        ),
        Variant("dd10_half", account_dd_half_at=-0.10),
        Variant("mkt_half_dd10_half", market_rule="half_prev_up45", account_dd_half_at=-0.10),
        Variant("mkt_tier_dd10_half", market_rule="tier_prev_up35_45", account_dd_half_at=-0.10),
        Variant("stop050_h5_mkt_half", stop_loss_pct=-0.050, stop_max_days=5, market_rule="half_prev_up45"),
        Variant("stop050_h5_dd10_half", stop_loss_pct=-0.050, stop_max_days=5, account_dd_half_at=-0.10),
        Variant(
            "stop050_h5_mkt_half_dd10_half",
            stop_loss_pct=-0.050,
            stop_max_days=5,
            market_rule="half_prev_up45",
            account_dd_half_at=-0.10,
        ),
        Variant(
            "stop050_h5_mkt_tier_dd10_half",
            stop_loss_pct=-0.050,
            stop_max_days=5,
            market_rule="tier_prev_up35_45",
            account_dd_half_at=-0.10,
        ),
        Variant("stop020_mkt_half_prev_up45", stop_loss_pct=-0.020, stop_min_days=0, market_rule="half_prev_up45"),
        Variant("stop015_mkt_half_prev_up45", stop_loss_pct=-0.015, stop_min_days=0, market_rule="half_prev_up45"),
        Variant("stop010_mkt_half_prev_up45", stop_loss_pct=-0.010, stop_min_days=0, market_rule="half_prev_up45"),
        Variant("stop005_mkt_half_prev_up45", stop_loss_pct=-0.005, stop_min_days=0, market_rule="half_prev_up45"),
        Variant("stop020_mkt_tier_prev_up35_45", stop_loss_pct=-0.020, stop_min_days=0, market_rule="tier_prev_up35_45"),
        Variant("score_weight", weight_mode="score"),
        Variant("lowrisk_weight", weight_mode="low_risk"),
        Variant("score_lowrisk_weight", weight_mode="score_low_risk"),
        Variant(
            "stop020_mkt_half_score_lowrisk",
            stop_loss_pct=-0.020,
            stop_min_days=0,
            market_rule="half_prev_up45",
            weight_mode="score_low_risk",
        ),
    ]


def _base_constraints() -> PortfolioConstraints:
    live = archived_constraints()
    return replace(
        live,
        candidate_risk_max_rank_pct=0.55,
        core_risk_max_rank_pct=0.45,
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
    )


def _constraints_for_variant(variant: Variant) -> PortfolioConstraints:
    constraints = _base_constraints()
    holding_policy = constraints.holding_policy
    if variant.target_hold_days is not None:
        holding_policy = replace(holding_policy, target_hold_days=int(variant.target_hold_days))
    if variant.max_hold_days is not None:
        holding_policy = replace(holding_policy, max_hold_days=int(variant.max_hold_days))
    if variant.force_exit_after_max_hold_days is not None:
        holding_policy = replace(
            holding_policy,
            force_exit_after_max_hold_days=bool(variant.force_exit_after_max_hold_days),
        )
    return replace(constraints, holding_policy=holding_policy)


def archived_constraints() -> PortfolioConstraints:
    return PortfolioConstraints(
        target_positions=12,
        hard_max_positions=15,
        max_initial_entries=12,
        max_new_entries_per_day=4,
        max_industry_names=3,
        max_unknown_industry_names=1,
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


def _load_scored_candidates(con: duckdb.DuckDBPyConnection, year: int) -> pd.DataFrame:
    start = f"{year}-01-01"
    end = f"{year}-12-31"
    fold_id = PREDICTION_FOLD_ID or f"wf_{year}"
    sql = """
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
    """
    frame = con.execute(sql, [RUN_ID, fold_id, SCORE_VERSION, start, end]).fetchdf()
    if frame.empty:
        raise ValueError(f"no candidates loaded for {year}")
    scored = archived_adv_score(frame)
    scored["run_id"] = RUN_ID
    scored["fold_id"] = fold_id
    return scored


def _load_bars(con: duckdb.DuckDBPyConnection, year: int) -> pd.DataFrame:
    start = f"{year}-01-01"
    end = f"{year}-12-31"
    sql = """
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
    """
    frame = con.execute(sql, [start, end]).fetchdf()
    if frame.empty:
        raise ValueError(f"no bars loaded for {year}")
    return frame


def _load_market_state(con: duckdb.DuckDBPyConnection, start_year: int, end_year: int) -> dict[str, dict[str, float]]:
    start = f"{start_year}-01-01"
    end = f"{end_year}-12-31"
    sql = """
        select
            trade_date,
            avg(case when close > prev_close then 1.0 else 0.0 end) as up_ratio,
            avg(close / nullif(prev_close, 0) - 1.0) as avg_ret
        from ml_tradeability_daily
        where trade_date between ? and ?
          and close > 0
          and prev_close > 0
          and coalesce(is_st, false) = false
          and coalesce(is_paused, false) = false
          and coalesce(is_bse, false) = false
          and coalesce(adv20_amount, 0) >= 10000000
        group by trade_date
        order by trade_date
    """
    daily = con.execute(sql, [start, end]).fetchdf()
    daily["prev_up_ratio"] = daily["up_ratio"].shift(1)
    daily["prev_avg_ret"] = daily["avg_ret"].shift(1)
    return {
        str(row.trade_date): {
            "up_ratio": float(row.up_ratio),
            "avg_ret": float(row.avg_ret),
            "prev_up_ratio": float(row.prev_up_ratio) if pd.notna(row.prev_up_ratio) else 1.0,
            "prev_avg_ret": float(row.prev_avg_ret) if pd.notna(row.prev_avg_ret) else 0.0,
        }
        for row in daily.itertuples(index=False)
    }


def run_research_backtest(
    scored_candidates: pd.DataFrame,
    bars: pd.DataFrame,
    market_state: dict[str, dict[str, float]],
    variant: Variant,
    year: int,
    execution: ExecutionConfig | None = None,
) -> ResearchResult:
    constraints = _constraints_for_variant(variant)
    execution = execution or ExecutionConfig()
    config = BacktestConfig(
        initial_cash=1_000_000.0,
        portfolio_id=f"{BASE_PORTFOLIO_ID}_{variant.name}_{year}",
        execution=execution,
        decision_dates=sorted(scored_candidates["trade_date"].dropna().unique()),
    )
    scored = scored_candidates.copy()
    candidates_by_date = _group_by_trade_date(scored)
    bars_sorted = bars.sort_values(["code", "trade_date"]).copy()
    bars_sorted.attrs["_vpa_sorted_by_code_date"] = True
    bar_index = BarDataIndex.from_bars(bars_sorted)
    trading_dates = sorted(bars_sorted["trade_date"].dropna().unique())
    decision_dates = _date_key_set(scored["trade_date"].dropna().unique())
    daily_bars = _daily_bar_lookup(bars_sorted)

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
        todays_orders = [order for order in pending_orders if order.get("sim_date") == date]
        pending_orders = [order for order in pending_orders if order.get("sim_date") != date]
        turnover_value = 0.0
        for order in sorted(todays_orders, key=_execution_priority):
            if order.get("status") != "filled":
                continue
            qty = float(order["qty"])
            fill_px = float(order["fill_px"])
            if order.get("side") == "buy":
                qty = _cash_limited_buy_qty_with_fees(qty, fill_px, cash, config.execution)
                if qty <= 1e-12:
                    order["qty"] = 0.0
                    order["fill_px"] = None
                    order["status"] = "rejected"
                    order["reason"] = "insufficient_cash"
                    continue
                order["qty"] = qty
            value = qty * fill_px
            commission, stamp_duty = _execution_fees(value, str(order.get("side")), config.execution)
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

        _advance_trading_holding_days(position_meta, str(date))
        _update_position_path_stats(positions, position_meta, daily_bars, date)
        gross = _position_value(positions, bar_index, date)
        nav = cash + gross
        peak_nav = max(peak_nav, nav)
        account_drawdown = nav / peak_nav - 1.0 if peak_nav else 0.0
        exposure_scalar = _exposure_scalar(variant, str(date), market_state, account_drawdown)
        nav_rows.append(
            {
                "run_id": RUN_ID,
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
                    "run_id": RUN_ID,
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

        date_key = _date_key(date)
        if date_key in decision_dates:
            pending_codes = {str(order["code"]) for order in pending_orders}
            day_candidates = candidates_by_date.get(date_key)
            if day_candidates is None:
                day_candidates = pd.DataFrame(columns=scored.columns)
            else:
                day_candidates = day_candidates.copy()
            holdings = _fixed_holdings_frame(positions, position_meta)
            targets = construct_portfolio_targets_v2(
                day_candidates,
                constraints,
                config.portfolio_id,
                current_holdings=holdings,
                run_id=RUN_ID,
                fold_id=f"wf_{year}",
                score_version=variant.name,
            )
            targets = _enrich_targets_from_candidates(targets, day_candidates)
            targets = _apply_profit_protection(targets, holdings, date, variant, config.portfolio_id)
            targets = _apply_selective_not_candidate_grace(targets, holdings, variant)
            targets = _apply_stop_loss(targets, holdings, bar_index, date, variant, config.portfolio_id)
            current_weights = _current_position_weights(positions, bar_index, date, nav)
            targets = _allocate_variant_weights(targets, variant, constraints, exposure_scalar, current_weights)
            diagnostics = get_portfolio_diagnostics(targets)
            if not diagnostics.empty:
                diagnostics = diagnostics.copy()
                diagnostics["period"] = year
                diagnostics["variant"] = variant.name
                diagnostics["target_exposure_scalar"] = exposure_scalar
                diagnostic_frames.append(diagnostics)
            orders = simulate_rebalance_orders(
                targets,
                bars_sorted,
                _positions_frame(positions, position_meta),
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
                    record["run_id"] = RUN_ID
                    record["fold_id"] = f"wf_{year}"
                    record["strategy_id"] = variant.name
                    record["score_version"] = variant.name
                    next_order_seq += 1
                all_orders.extend(records)
                pending_orders.extend(records)

    orders_frame = _with_realized_returns(pd.DataFrame(all_orders))
    nav_frame = pd.DataFrame(nav_rows)
    diagnostics_frame = pd.concat(diagnostic_frames, ignore_index=True) if diagnostic_frames else pd.DataFrame()
    return ResearchResult(
        variant=variant,
        year=year,
        orders=orders_frame,
        nav=nav_frame,
        diagnostics=diagnostics_frame,
    )


def _enrich_targets_from_candidates(targets: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    if targets.empty or candidates.empty:
        return targets
    attrs = dict(targets.attrs)
    enrich_cols = [
        "trade_date",
        "code",
        "trade_score_v2",
        "absolute_rank_pct",
        "active_rank_pct",
        "risk_rank_pct",
        "risk_prob",
        "full_prediction_pool_adv_pct",
        "adv20_amount",
    ]
    available = [column for column in enrich_cols if column in candidates.columns]
    if len(available) <= 2:
        return targets
    out = targets.merge(
        candidates[available].drop_duplicates(["trade_date", "code"]),
        on=["trade_date", "code"],
        how="left",
        suffixes=("", "_candidate"),
    )
    for column in available:
        if column in {"trade_date", "code"}:
            continue
        candidate_column = f"{column}_candidate"
        if candidate_column in out:
            if column not in out:
                out[column] = out[candidate_column]
            else:
                out[column] = out[column].fillna(out[candidate_column])
            out = out.drop(columns=[candidate_column])
    out.attrs.update(attrs)
    return out


def _apply_profit_protection(
    targets: pd.DataFrame,
    holdings: pd.DataFrame,
    date: object,
    variant: Variant,
    portfolio_id: str,
) -> pd.DataFrame:
    if not variant.profit_protect or holdings.empty:
        return targets
    attrs = dict(targets.attrs)
    out = targets.copy()
    if out.empty:
        out = pd.DataFrame(columns=TARGET_COLUMNS)
    generated_at = pd.Timestamp.now("UTC").isoformat()
    target_by_code = _target_index_by_code(out)
    protect_rows: list[dict[str, object]] = []
    for row in holdings.itertuples(index=False):
        code = str(row.code)
        holding_days = int(getattr(row, "holding_days", 0) or 0)
        if holding_days < variant.profit_protect_min_days:
            continue
        max_gain = _float_or_none(getattr(row, "max_high_ret", None))
        if max_gain is None:
            max_gain = _float_or_none(getattr(row, "max_close_ret", None))
        current_ret = _float_or_none(getattr(row, "current_close_ret", None))
        if max_gain is None or current_ret is None:
            continue
        if max_gain < variant.profit_protect_min_gain or current_ret > variant.profit_protect_exit_below:
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
                "trade_date": str(date),
                "portfolio_id": portfolio_id,
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
                "shares": getattr(row, "shares", getattr(row, "position_qty", None)),
                "holding_days": holding_days,
                "entry_trade_score": getattr(row, "entry_trade_score", None),
                "latest_trade_score": getattr(row, "latest_trade_score", getattr(row, "entry_trade_score", None)),
                "generated_at": generated_at,
            }
        )
    if protect_rows:
        out = pd.concat([out, pd.DataFrame(protect_rows)], ignore_index=True)
    out.attrs.update(attrs)
    return _ordered_target_columns(out)


def _apply_selective_not_candidate_grace(
    targets: pd.DataFrame,
    holdings: pd.DataFrame,
    variant: Variant,
) -> pd.DataFrame:
    if not variant.selective_not_candidate_grace or targets.empty or holdings.empty:
        return targets
    attrs = dict(targets.attrs)
    out = targets.copy()
    holding_by_code = {str(row.code): row for row in holdings.itertuples(index=False)}
    if "exit_reason" not in out or "signal_action" not in out:
        return targets
    for idx, row in out.iterrows():
        if str(row.get("signal_action")) != "sell" or row.get("exit_reason") != "not_candidate_after_target_days":
            continue
        code = str(row.get("code"))
        holding = holding_by_code.get(code)
        if holding is None or not _eligible_for_not_candidate_grace(row, holding, variant):
            continue
        out.at[idx, "target_weight"] = 0.0
        out.at[idx, "entry_reason"] = getattr(holding, "entry_reason", row.get("entry_reason"))
        out.at[idx, "signal_action"] = "hold"
        out.at[idx, "hold_reason"] = "grace_not_candidate_lowadv"
        out.at[idx, "exit_reason"] = None
        out.at[idx, "sell_blocked_reason"] = None
        out.at[idx, "shares"] = getattr(holding, "shares", row.get("shares", None))
        out.at[idx, "entry_date"] = getattr(holding, "entry_date", row.get("entry_date", None))
        out.at[idx, "entry_price"] = getattr(holding, "entry_price", row.get("entry_price", None))
        out.at[idx, "holding_days"] = getattr(holding, "holding_days", row.get("holding_days", None))
    out.attrs.update(attrs)
    return _ordered_target_columns(out)


def _eligible_for_not_candidate_grace(row: pd.Series, holding: object, variant: Variant) -> bool:
    holding_days = int(getattr(holding, "holding_days", 0) or 0)
    if holding_days >= int(variant.grace_max_hold_days):
        return False
    trade_score = _float_or_none(row.get("trade_score_v2", row.get("trade_score")))
    if trade_score is None or trade_score < variant.grace_min_trade_score:
        return False
    risk_rank = _float_or_none(row.get("risk_rank_pct"))
    if risk_rank is None or risk_rank > variant.grace_max_risk_rank_pct:
        return False
    adv_pct = _float_or_none(row.get("full_prediction_pool_adv_pct"))
    low_adv_score = 1.0 - adv_pct if adv_pct is not None else None
    if low_adv_score is None or low_adv_score < variant.grace_min_low_adv_score:
        return False
    current_ret = _float_or_none(getattr(holding, "current_close_ret", None))
    if (
        current_ret is None
        or current_ret < variant.grace_min_current_ret
        or current_ret > variant.grace_max_current_ret
    ):
        return False
    prior_gain = _float_or_none(getattr(holding, "max_high_ret", None))
    if prior_gain is None:
        prior_gain = _float_or_none(getattr(holding, "max_close_ret", None))
    if prior_gain is not None and prior_gain >= variant.grace_max_prior_gain:
        return False
    return True


def _target_index_by_code(targets: pd.DataFrame) -> dict[str, object]:
    if targets.empty or "code" not in targets:
        return {}
    return {str(row["code"]): idx for idx, row in targets.iterrows()}


def _ordered_target_columns(targets: pd.DataFrame) -> pd.DataFrame:
    if targets.empty:
        return targets
    return targets[[column for column in TARGET_COLUMNS if column in targets.columns] + [column for column in targets.columns if column not in TARGET_COLUMNS]]


def _apply_stop_loss(
    targets: pd.DataFrame,
    holdings: pd.DataFrame,
    bar_index: BarDataIndex,
    date: object,
    variant: Variant,
    portfolio_id: str,
) -> pd.DataFrame:
    if variant.stop_loss_pct is None or holdings.empty:
        return targets
    stop_rows = []
    stop_codes: set[str] = set()
    generated_at = pd.Timestamp.now("UTC").isoformat()
    for row in holdings.itertuples(index=False):
        code = str(row.code)
        entry_price = _float_or_none(getattr(row, "entry_price", None))
        if entry_price is None or entry_price <= 0:
            continue
        holding_days = int(getattr(row, "holding_days", 0) or 0)
        if holding_days < variant.stop_min_days:
            continue
        if variant.stop_max_days is not None and holding_days > int(variant.stop_max_days):
            continue
        close = bar_index.close_at_or_before(code, date)
        if close <= 0:
            continue
        current_ret = close / entry_price - 1.0
        if current_ret > float(variant.stop_loss_pct):
            continue
        stop_codes.add(code)
        stop_rows.append(
            {
                "trade_date": str(date),
                "portfolio_id": portfolio_id,
                "code": code,
                "target_weight": 0.0,
                "rank_n": None,
                "trade_score": getattr(row, "latest_trade_score", getattr(row, "entry_trade_score", None)),
                "entry_reason": getattr(row, "entry_reason", None),
                "signal_action": "sell",
                "hold_reason": None,
                "exit_reason": _stop_loss_exit_reason(variant),
                "sell_blocked_reason": None,
                "entry_date": getattr(row, "entry_date", None),
                "entry_price": entry_price,
                "shares": getattr(row, "shares", getattr(row, "position_qty", None)),
                "holding_days": holding_days,
                "latest_trade_score": getattr(row, "latest_trade_score", getattr(row, "entry_trade_score", None)),
                "generated_at": generated_at,
            }
        )
    if not stop_rows:
        return targets
    attrs = dict(targets.attrs)
    base = targets.copy()
    if not base.empty and "code" in base:
        base = base[~base["code"].astype(str).isin(stop_codes)].copy()
    out = pd.concat([base, pd.DataFrame(stop_rows)], ignore_index=True)
    out = out[[column for column in TARGET_COLUMNS if column in out.columns] + [column for column in out.columns if column not in TARGET_COLUMNS]]
    out.attrs.update(attrs)
    return out


def _stop_loss_exit_reason(variant: Variant) -> str:
    reason = f"stop_loss_{abs(float(variant.stop_loss_pct)):.3f}"
    if variant.stop_max_days is not None:
        reason = f"{reason}_h{int(variant.stop_max_days)}"
    return reason


def _allocate_variant_weights(
    targets: pd.DataFrame,
    variant: Variant,
    constraints: PortfolioConstraints,
    exposure_scalar: float,
    current_weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    attrs = dict(targets.attrs)
    out = targets.copy()
    out.attrs.update(attrs)
    if out.empty:
        return out
    active_mask = pd.Series(True, index=out.index)
    if "signal_action" in out:
        active_mask &= out["signal_action"].astype(str) != "sell"
    active_count = int(active_mask.sum())
    out["target_weight"] = 0.0
    if active_count == 0:
        out.attrs.update(attrs)
        return out

    exposure_scalar = max(min(exposure_scalar, 1.0), 0.0)
    base_unit_weight = _base_equal_weight(active_count)
    if variant.exposure_mode == "entry_throttle" and exposure_scalar < 1.0:
        out = _freeze_existing_and_throttle_entries(out, active_mask, exposure_scalar, base_unit_weight, current_weights)
        out.attrs.update(attrs)
        return out
    if variant.weight_mode == "equal":
        weights = pd.Series(base_unit_weight, index=out.index[active_mask])
        total = float(weights.sum())
        if total > 0:
            weights = weights / total * exposure_scalar
        out.loc[active_mask, "target_weight"] = weights.values
        out.attrs.update(attrs)
        return out

    factors = _weight_factors(out.loc[active_mask].copy(), variant.weight_mode)
    weights = _bounded_normalized_weights(
        factors,
        total_weight=exposure_scalar,
        max_weight=1.0,
        min_weight=0.0,
    )
    out.loc[active_mask, "target_weight"] = weights.values
    out.attrs.update(attrs)
    return out


def _current_position_weights(
    positions: dict[str, float],
    bar_index: BarDataIndex,
    date: object,
    nav: float,
) -> dict[str, float]:
    if nav <= 0:
        return {}
    weights = {}
    for code, qty in positions.items():
        px = bar_index.close_at_or_before(code, date)
        if px <= 0:
            continue
            weights[str(code)] = max(float(qty) * float(px) / float(nav), 0.0)
    return weights


def _daily_bar_lookup(bars: pd.DataFrame) -> dict[tuple[str, str], dict[str, float]]:
    if bars.empty:
        return {}
    lookup: dict[tuple[str, str], dict[str, float]] = {}
    for row in bars.itertuples(index=False):
        lookup[(str(row.code), _date_key(getattr(row, "trade_date")))] = {
            "close": float(getattr(row, "close", 0.0) or 0.0),
            "high": float(getattr(row, "high", 0.0) or 0.0),
            "low": float(getattr(row, "low", 0.0) or 0.0),
        }
    return lookup


def _update_position_path_stats(
    positions: dict[str, float],
    position_meta: dict[str, dict[str, object]],
    daily_bars: dict[tuple[str, str], dict[str, float]],
    date: object,
) -> None:
    date_key = _date_key(date)
    for code in positions:
        meta = position_meta.get(code)
        if meta is None:
            continue
        entry_price = _float_or_none(meta.get("entry_price"))
        if entry_price is None or entry_price <= 0.0:
            continue
        bar = daily_bars.get((str(code), date_key))
        if not bar:
            continue
        close_ret = bar["close"] / entry_price - 1.0 if bar["close"] > 0.0 else None
        high_ret = bar["high"] / entry_price - 1.0 if bar["high"] > 0.0 else close_ret
        low_ret = bar["low"] / entry_price - 1.0 if bar["low"] > 0.0 else close_ret
        if close_ret is not None:
            meta["current_close_ret"] = close_ret
            meta["max_close_ret"] = max(float(meta.get("max_close_ret", close_ret) or close_ret), close_ret)
        if high_ret is not None:
            meta["max_high_ret"] = max(float(meta.get("max_high_ret", high_ret) or high_ret), high_ret)
        if low_ret is not None:
            meta["min_low_ret"] = min(float(meta.get("min_low_ret", low_ret) or low_ret), low_ret)


def _freeze_existing_and_throttle_entries(
    targets: pd.DataFrame,
    active_mask: pd.Series,
    exposure_scalar: float,
    base_unit_weight: float,
    current_weights: dict[str, float] | None,
) -> pd.DataFrame:
    out = targets.copy()
    current_weights = current_weights or {}
    codes = out["code"].astype(str) if "code" in out else pd.Series("", index=out.index)
    held_mask = active_mask & codes.isin(current_weights)
    entry_mask = active_mask & ~held_mask
    if held_mask.any():
        out.loc[held_mask, "target_weight"] = codes.loc[held_mask].map(current_weights).fillna(0.0).values
    if entry_mask.any() and exposure_scalar > 0:
        out.loc[entry_mask, "target_weight"] = base_unit_weight * exposure_scalar
    return out


def _base_equal_weight(active_count: int, min_weight: float = 0.05, max_weight: float = 0.10) -> float:
    if active_count <= 0:
        return 0.0
    equal_weight = 1.0 / active_count
    if active_count * min_weight <= 1.0:
        return min(max(equal_weight, min_weight), max_weight)
    return min(equal_weight, max_weight)


def _weight_factors(active: pd.DataFrame, mode: str) -> pd.Series:
    index = active.index
    score = pd.to_numeric(active.get("trade_score", active.get("trade_score_v2", pd.Series(0.0, index=index))), errors="coerce")
    score_rank = score.rank(method="average", pct=True).fillna(0.5)
    risk = pd.to_numeric(active.get("risk_rank_pct", pd.Series(0.5, index=index)), errors="coerce").fillna(0.5)
    low_risk = (1.0 - risk).clip(0.0, 1.0)
    low_risk_rank = low_risk.rank(method="average", pct=True).fillna(0.5)
    if mode == "score":
        raw = score_rank
    elif mode == "low_risk":
        raw = low_risk_rank
    elif mode == "score_low_risk":
        raw = 0.65 * score_rank + 0.35 * low_risk_rank
    else:
        raw = pd.Series(0.5, index=index)
    return (0.5 + raw).clip(0.5, 1.5)


def _bounded_normalized_weights(
    factors: pd.Series,
    *,
    total_weight: float,
    max_weight: float,
    min_weight: float,
) -> pd.Series:
    if factors.empty or total_weight <= 0:
        return pd.Series(0.0, index=factors.index)
    positive = factors.clip(lower=0.01)
    weights = positive / positive.sum() * total_weight
    if total_weight / len(weights) < min_weight:
        min_weight = 0.0
    for _ in range(8):
        capped = weights.clip(lower=min_weight, upper=max_weight)
        fixed = (capped == min_weight) | (capped == max_weight)
        free = ~fixed
        remainder = total_weight - capped[fixed].sum()
        if not free.any() or abs(capped.sum() - total_weight) < 1e-9:
            return capped
        free_factors = positive[free]
        weights = capped
        weights.loc[free] = free_factors / free_factors.sum() * max(remainder, 0.0)
    return weights.clip(lower=0.0, upper=max_weight)


def _exposure_scalar(
    variant: Variant,
    date: str,
    market_state: dict[str, dict[str, float]],
    account_drawdown: float,
) -> float:
    scalar = 1.0
    state = market_state.get(date, {})
    prev_up_ratio = float(state.get("prev_up_ratio", 1.0))
    if variant.market_zero_below is not None or variant.market_half_below is not None:
        if variant.market_zero_below is not None and prev_up_ratio < float(variant.market_zero_below):
            scalar = min(scalar, 0.0)
        elif variant.market_half_below is not None and prev_up_ratio < float(variant.market_half_below):
            scalar = min(scalar, 0.5)
    elif variant.market_rule == "half_prev_up45" and prev_up_ratio < 0.45:
        scalar = min(scalar, 0.5)
    elif variant.market_rule == "tier_prev_up35_45":
        if prev_up_ratio < 0.35:
            scalar = min(scalar, 0.0)
        elif prev_up_ratio < 0.45:
            scalar = min(scalar, 0.5)
    if variant.account_dd_half_at is not None and account_drawdown <= variant.account_dd_half_at:
        scalar = min(scalar, 0.5)
    return scalar


def _cash_limited_buy_qty_with_fees(
    qty: float,
    fill_px: float,
    cash: float,
    execution: ExecutionConfig,
) -> float:
    if fill_px <= 0.0 or cash <= 0.0:
        return 0.0
    gross_multiplier = 1.0 + execution.commission_bps / 10000.0
    affordable_qty = min(qty, cash / (fill_px * gross_multiplier))
    if not execution.allow_fractional_shares:
        affordable_qty = int(affordable_qty // execution.a_share_lot_size) * execution.a_share_lot_size
    return max(affordable_qty, 0.0)


def _execution_fees(value: float, side: str, execution: ExecutionConfig) -> tuple[float, float]:
    commission = value * execution.commission_bps / 10000.0
    stamp_duty = value * execution.stamp_duty_bps / 10000.0 if side == "sell" else 0.0
    return commission, stamp_duty


def _with_realized_returns(orders: pd.DataFrame) -> pd.DataFrame:
    if orders.empty:
        return orders
    out = orders.copy()
    mask = (out["side"] == "sell") & (out["status"] == "filled")
    out["realized_ret"] = pd.NA
    out.loc[mask, "realized_ret"] = (
        pd.to_numeric(out.loc[mask, "fill_px"], errors="coerce")
        / pd.to_numeric(out.loc[mask, "entry_price"], errors="coerce")
        - 1.0
    )
    out["realized_pnl"] = pd.NA
    out.loc[mask, "realized_pnl"] = (
        (
            pd.to_numeric(out.loc[mask, "fill_px"], errors="coerce")
            - pd.to_numeric(out.loc[mask, "entry_price"], errors="coerce")
        )
        * pd.to_numeric(out.loc[mask, "qty"], errors="coerce")
    )
    return out


def _summarize_results(results: list[ResearchResult]) -> pd.DataFrame:
    rows = []
    for variant_name, group in _group_results(results).items():
        nav_by_year = {result.year: result.nav for result in group}
        yearly_returns = []
        max_drawdown = 0.0
        avg_exposure_values = []
        for year, nav in nav_by_year.items():
            if nav.empty:
                continue
            yearly_return = float(nav["nav"].iloc[-1] / nav["nav"].iloc[0] - 1.0)
            yearly_returns.append(yearly_return)
            max_drawdown = min(max_drawdown, _max_drawdown(nav["nav"]))
            avg_exposure_values.append(float(pd.to_numeric(nav["gross_exposure"], errors="coerce").mean()))
        compounded_return = _compound_returns(yearly_returns)
        annual_return = (1.0 + compounded_return) ** (1.0 / len(yearly_returns)) - 1.0 if yearly_returns else 0.0
        orders = pd.concat([result.orders for result in group if not result.orders.empty], ignore_index=True) if group else pd.DataFrame()
        trade = _trade_metrics(orders)
        rows.append(
            {
                "variant": variant_name,
                "years": len(yearly_returns),
                "total_return": compounded_return,
                "annual_return": annual_return,
                "max_drawdown": max_drawdown,
                "avg_exposure": float(pd.Series(avg_exposure_values).mean()) if avg_exposure_values else 0.0,
                **trade,
            }
        )
    return pd.DataFrame(rows)


def _summarize_years(results: list[ResearchResult]) -> pd.DataFrame:
    rows = []
    for result in results:
        nav = result.nav
        orders = result.orders
        yearly_return = float(nav["nav"].iloc[-1] / nav["nav"].iloc[0] - 1.0) if not nav.empty else 0.0
        rows.append(
            {
                "variant": result.variant.name,
                "year": result.year,
                "total_return": yearly_return,
                "max_drawdown": _max_drawdown(nav["nav"]) if not nav.empty else 0.0,
                "avg_exposure": float(pd.to_numeric(nav["gross_exposure"], errors="coerce").mean()) if not nav.empty else 0.0,
                **_trade_metrics(orders),
            }
        )
    return pd.DataFrame(rows)


def _trade_metrics(orders: pd.DataFrame) -> dict[str, float]:
    if orders.empty or "realized_ret" not in orders:
        return {
            "trade_count": 0.0,
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "loss_to_win": 0.0,
            "payoff": 0.0,
            "profit_factor": 0.0,
        }
    rets = pd.to_numeric(orders.loc[(orders["side"] == "sell") & (orders["status"] == "filled"), "realized_ret"], errors="coerce").dropna()
    wins = rets[rets > 0]
    losses = rets[rets < 0]
    avg_win = float(wins.mean()) if not wins.empty else 0.0
    avg_loss = float(losses.mean()) if not losses.empty else 0.0
    gross_win = float(wins.sum()) if not wins.empty else 0.0
    gross_loss = float(losses.sum()) if not losses.empty else 0.0
    return {
        "trade_count": float(len(rets)),
        "win_rate": float((rets > 0).mean()) if not rets.empty else 0.0,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "loss_to_win": abs(avg_loss) / avg_win if avg_win else 0.0,
        "payoff": avg_win / abs(avg_loss) if avg_loss else 0.0,
        "profit_factor": gross_win / abs(gross_loss) if gross_loss else 0.0,
    }


def _group_results(results: list[ResearchResult]) -> dict[str, list[ResearchResult]]:
    grouped: dict[str, list[ResearchResult]] = {}
    for result in results:
        grouped.setdefault(result.variant.name, []).append(result)
    return grouped


def _concat_result_frames(results: Iterable[ResearchResult], attr: str) -> pd.DataFrame:
    frames = [getattr(result, attr) for result in results if not getattr(result, attr).empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _compound_returns(returns: list[float]) -> float:
    value = 1.0
    for ret in returns:
        value *= 1.0 + ret
    return value - 1.0


def _max_drawdown(nav: pd.Series) -> float:
    values = pd.to_numeric(nav, errors="coerce").dropna()
    if values.empty:
        return 0.0
    drawdown = values / values.cummax() - 1.0
    return float(drawdown.min())


def _float_or_none(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


if __name__ == "__main__":
    main()
