from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


DEFAULT_VPA_DB = "outputs/vpa.duckdb"
DEFAULT_SHARED_ML_DB = "outputs/ml/ml_ret5_alpha_risk_20260619.duckdb"
DEFAULT_OUT_DB = "outputs/limit_hit_research/limit_hit_research.duckdb"
DEFAULT_OUT_DIR = "outputs/limit_hit_research/reports"
RUN_ID = "limit_hit_v1"


@dataclass(frozen=True)
class LimitHitResearchConfig:
    train_start: str = "2015-01-05"
    start_year: int = 2020
    end_year: int = 2025
    min_adv20_amount: float = 20_000_000.0
    min_amount: float = 20_000_000.0
    candidate_min_ret: float = 0.02
    entry_min_ret: float = 0.10
    exclude_bse: bool = True
    exclude_st: bool = True
    negative_sample_ratio: float = 8.0
    random_state: int = 20260622
    alpha_rounds: int = 160
    risk_rounds: int = 120
    max_positions: int = 1
    max_position_weight: float = 0.69
    initial_cash: float = 1_000_000.0
    slippage_bps: float = 10.0
    commission_bps: float = 3.0
    stamp_duty_bps: float = 5.0
    min_probability: float = 0.35
    max_risk_prob: float = 0.10
    risk_weight: float = 0.35
    max_drawdown_stop: float = -0.30
    drawdown_reduce_at: float = -0.20
    reduced_max_positions: int = 2
    limit_hit_extra_hold_days: int = 1
    limit_success_mode: str = "close"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vpa-db", default=DEFAULT_VPA_DB)
    parser.add_argument("--shared-ml-db", default=DEFAULT_SHARED_ML_DB)
    parser.add_argument("--out-db", default=DEFAULT_OUT_DB)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--start-year", type=int, default=2020)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--train-start", default="2015-01-05")
    parser.add_argument("--min-adv20-amount", type=float, default=20_000_000.0)
    parser.add_argument("--min-amount", type=float, default=20_000_000.0)
    parser.add_argument("--candidate-min-ret", type=float, default=0.02)
    parser.add_argument("--entry-min-ret", type=float, default=0.10)
    parser.add_argument("--max-positions", type=int, default=1)
    parser.add_argument("--max-position-weight", type=float, default=0.69)
    parser.add_argument("--min-probability", type=float, default=0.35)
    parser.add_argument("--max-risk-prob", type=float, default=0.10)
    parser.add_argument("--risk-weight", type=float, default=0.35)
    parser.add_argument("--limit-hit-extra-hold-days", type=int, default=1)
    parser.add_argument("--limit-success-mode", choices=["touch", "close"], default="close")
    parser.add_argument("--smoke", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = LimitHitResearchConfig(
        train_start=args.train_start,
        start_year=args.start_year,
        end_year=args.end_year,
        min_adv20_amount=args.min_adv20_amount,
        min_amount=args.min_amount,
        candidate_min_ret=args.candidate_min_ret,
        entry_min_ret=args.entry_min_ret,
        max_positions=args.max_positions,
        max_position_weight=args.max_position_weight,
        min_probability=args.min_probability,
        max_risk_prob=args.max_risk_prob,
        risk_weight=args.risk_weight,
        limit_hit_extra_hold_days=args.limit_hit_extra_hold_days,
        limit_success_mode=args.limit_success_mode,
    )
    run_limit_hit_research(
        vpa_db=Path(args.vpa_db),
        shared_ml_db=Path(args.shared_ml_db),
        out_db=Path(args.out_db),
        out_dir=Path(args.out_dir),
        config=config,
        smoke=bool(args.smoke),
    )


def run_limit_hit_research(
    *,
    vpa_db: Path,
    shared_ml_db: Path,
    out_db: Path,
    out_dir: Path,
    config: LimitHitResearchConfig,
    smoke: bool = False,
) -> None:
    out_db.parent.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(out_db))
    try:
        _init_research_db(con)
        con.execute(f"attach '{_duckdb_path_literal(vpa_db)}' as vpa_db (read_only)")
        con.execute(f"attach '{_duckdb_path_literal(shared_ml_db)}' as shared_ml (read_only)")
        dataset = load_limit_hit_dataset(con, config, smoke=smoke)
        if dataset.empty:
            raise RuntimeError("limit-hit dataset is empty")
        predictions, model_rows = walkforward_predict(dataset, config)
        backtest_bars = load_backtest_bars(con, predictions)
        result = run_limit_hit_backtest(
            predictions,
            backtest_bars,
            initial_cash=config.initial_cash,
            max_positions=config.max_positions,
            max_position_weight=config.max_position_weight,
            min_probability=config.min_probability,
            max_risk_prob=config.max_risk_prob,
            risk_weight=config.risk_weight,
            slippage_bps=config.slippage_bps,
            commission_bps=config.commission_bps,
            stamp_duty_bps=config.stamp_duty_bps,
            max_drawdown_stop=config.max_drawdown_stop,
            drawdown_reduce_at=config.drawdown_reduce_at,
            reduced_max_positions=config.reduced_max_positions,
            limit_hit_extra_hold_days=config.limit_hit_extra_hold_days,
            limit_success_mode=config.limit_success_mode,
            entry_min_ret=config.entry_min_ret,
        )
        manifest = {
            "run_id": RUN_ID,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "vpa_db": str(vpa_db),
            "shared_ml_db": str(shared_ml_db),
            "out_db": str(out_db),
            "config": asdict(config),
            "dataset_rows": int(len(dataset)),
            "prediction_rows": int(len(predictions)),
            "metrics": result["metrics"],
        }
        _replace_table(con, "lh_labels_daily", dataset)
        _replace_table(con, "lh_predictions_daily", predictions)
        _replace_table(con, "lh_orders", result["orders"])
        _replace_table(con, "lh_nav", result["nav"])
        _replace_table(con, "lh_yearly_metrics", pd.DataFrame(result["yearly_metrics"]))
        _replace_table(con, "lh_model_folds", pd.DataFrame(model_rows))
        _replace_table(con, "lh_run_manifest", pd.DataFrame([{"run_id": RUN_ID, "manifest_json": json.dumps(manifest, sort_keys=True)}]))
        dataset.head(5000).to_csv(out_dir / "limit_hit_dataset_sample.csv", index=False)
        predictions.to_csv(out_dir / "limit_hit_predictions.csv", index=False)
        result["orders"].to_csv(out_dir / "limit_hit_orders.csv", index=False)
        result["nav"].to_csv(out_dir / "limit_hit_nav.csv", index=False)
        pd.DataFrame(result["yearly_metrics"]).to_csv(out_dir / "limit_hit_yearly_metrics.csv", index=False)
        (out_dir / "limit_hit_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps(manifest["metrics"], indent=2, sort_keys=True))
        print(f"wrote {out_db} and {out_dir}")
    finally:
        con.close()


def load_limit_hit_dataset(
    con: duckdb.DuckDBPyConnection,
    config: LimitHitResearchConfig,
    *,
    smoke: bool = False,
) -> pd.DataFrame:
    end_date = f"{config.end_year}-12-31"
    smoke_start = f"{max(config.start_year - 1, int(config.train_start[:4]))}-01-01"
    smoke_end = f"{config.start_year}-03-31"
    smoke_clause = f"and t.trade_date >= '{smoke_start}' and t.trade_date <= '{smoke_end}'" if smoke else ""
    limit_clause = ""
    query = f"""
        with base as (
            select
                t.trade_date,
                t.code,
                t.open,
                t.high,
                t.low,
                t.close,
                t.prev_close,
                t.limit_up,
                t.limit_down,
                t.limit_up_pct,
                t.limit_down_pct,
                t.limit_band,
                t.amount,
                t.turnover_rate,
                t.adv20_amount,
                t.is_st,
                t.is_paused,
                t.is_bse,
                t.can_buy_next_open,
                t.can_sell_next_open,
                t.next_trade_date,
                t.next_open,
                t.next_limit_up,
                t.next_limit_down,
                n.high as next_high_actual,
                n.low as next_low_actual,
                n.close as next_close_actual,
                n.limit_up as next_limit_up_actual,
                n2.open as next2_open,
                n3.open as next3_open,
                st.market_score,
                st.sector_score,
                st.self_score,
                st.relative_strength_score,
                st.resonance_score,
                st.confidence,
                max(case when f.window_n = 5 then f.ret_pct end) as ret_5w,
                max(case when f.window_n = 10 then f.ret_pct end) as ret_10w,
                max(case when f.window_n = 20 then f.ret_pct end) as ret_20w,
                max(case when f.window_n = 60 then f.ret_pct end) as ret_60w,
                max(case when f.window_n = 5 then f.vol_rvol_n end) as vol_rvol_5,
                max(case when f.window_n = 20 then f.vol_rvol_n end) as vol_rvol_20,
                max(case when f.window_n = 5 then f.range_rvol_n end) as range_rvol_5,
                max(case when f.window_n = 20 then f.range_rvol_n end) as range_rvol_20,
                max(case when f.window_n = 20 then f.price_position_n end) as price_position_20,
                max(case when f.window_n = 60 then f.price_position_n end) as price_position_60,
                max(case when f.window_n = 20 then f.ma_slope_n end) as ma_slope_20,
                max(case when f.window_n = 60 then f.ma_slope_n end) as ma_slope_60,
                max(case when q.window_n = 20 then q.sequence_strength_score end) as sequence_strength_20,
                max(case when q.window_n = 60 then q.sequence_strength_score end) as sequence_strength_60
            from shared_ml.ml_tradeability_daily t
            left join shared_ml.ml_tradeability_daily n
              on n.trade_date = t.next_trade_date
             and n.code = t.code
            left join shared_ml.ml_tradeability_daily n2
              on n2.trade_date = n.next_trade_date
             and n2.code = t.code
            left join shared_ml.ml_tradeability_daily n3
              on n3.trade_date = n2.next_trade_date
             and n3.code = t.code
            left join vpa_db.vpa_structure_state st
              on st.date = t.trade_date
             and st.scope_type = 'stock'
             and st.scope_id = t.code
            left join vpa_db.vpa_features f
              on f.date = t.trade_date
             and f.scope_type = 'stock'
             and f.scope_id = t.code
             and f.window_n in (5, 10, 20, 60)
            left join vpa_db.vpa_sequence_stats q
              on q.date = t.trade_date
             and q.scope_type = 'stock'
             and q.scope_id = t.code
             and q.window_n in (20, 60)
            where t.trade_date >= ?
              and t.trade_date <= ?
              and coalesce(t.is_paused, false) = false
              and (? = false or coalesce(t.is_st, false) = false)
              and (? = false or coalesce(t.is_bse, false) = false)
              and coalesce(t.adv20_amount, 0) >= ?
              and coalesce(t.amount, 0) >= ?
              and (t.close / nullif(t.prev_close, 0) - 1.0) >= ?
              {smoke_clause}
            group by all
            order by t.trade_date, t.code
            {limit_clause}
        )
        select * from base
    """
    frame = con.execute(
        query,
        [
            config.train_start,
            end_date,
            config.exclude_st,
            config.exclude_bse,
            config.min_adv20_amount,
            config.min_amount,
            config.candidate_min_ret,
        ],
    ).fetchdf()
    return _add_labels_and_features(
        frame,
        limit_hit_extra_hold_days=config.limit_hit_extra_hold_days,
        limit_success_mode=config.limit_success_mode,
    )


def load_backtest_bars(con: duckdb.DuckDBPyConnection, predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame(columns=["trade_date", "code", "open", "high", "low", "close", "limit_up", "limit_down", "is_paused"])
    codes = pd.DataFrame({"code": sorted(predictions["code"].astype(str).unique())})
    con.register("_lh_backtest_codes", codes)
    try:
        start_date = str(predictions["trade_date"].min())
        max_entry_date = datetime.strptime(str(predictions["next_trade_date"].max())[:10], "%Y-%m-%d")
        end_date = (max_entry_date + timedelta(days=30)).date().isoformat()
        return con.execute(
            """
            select
                t.trade_date,
                t.code,
                t.open,
                t.high,
                t.low,
                t.close,
                t.limit_up,
                t.limit_down,
                t.is_paused
            from shared_ml.ml_tradeability_daily t
            join _lh_backtest_codes c using (code)
            where t.trade_date >= ?
              and t.trade_date <= ?
            order by t.code, t.trade_date
            """,
            [start_date, end_date],
        ).fetchdf()
    finally:
        con.unregister("_lh_backtest_codes")


def build_limit_hit_labels(
    bars: pd.DataFrame,
    *,
    limit_hit_extra_hold_days: int = 1,
    limit_success_mode: str = "touch",
) -> pd.DataFrame:
    ordered = bars.sort_values(["code", "trade_date"]).copy()
    grouped = ordered.groupby("code", sort=False)
    for column in ["trade_date", "open", "high", "low", "close", "limit_up", "limit_down", "is_paused"]:
        if column in ordered:
            ordered[f"next_{column}"] = grouped[column].shift(-1)
    ordered["next_trade_date"] = ordered["next_trade_date"].astype(object)
    ordered["hit_limit_next_day"] = _limit_success_mask(ordered, "next_", limit_success_mode).astype(int)
    ordered["next_open_ret"] = pd.to_numeric(ordered["next_open"], errors="coerce") / pd.to_numeric(ordered["close"], errors="coerce") - 1.0
    if "next_close" in ordered:
        ordered["next_close_ret_from_open"] = pd.to_numeric(ordered["next_close"], errors="coerce") / pd.to_numeric(ordered["next_open"], errors="coerce") - 1.0
    else:
        ordered["next_close_ret_from_open"] = np.nan
    if "next_low" in ordered:
        ordered["next_low_ret_from_open"] = pd.to_numeric(ordered["next_low"], errors="coerce") / pd.to_numeric(ordered["next_open"], errors="coerce") - 1.0
    else:
        ordered["next_low_ret_from_open"] = np.nan
    ordered["risk_bad_next_day"] = ((ordered["hit_limit_next_day"] == 0) & (ordered["next_low_ret_from_open"] <= -0.04)).astype(int)
    if "next_open" in ordered:
        ordered["next2_open"] = grouped["open"].shift(-2)
        ordered["next3_open"] = grouped["open"].shift(-3)
        miss_ret = pd.to_numeric(ordered["next2_open"], errors="coerce") / pd.to_numeric(ordered["next_open"], errors="coerce") - 1.0
        hit_exit = ordered["next2_open"] if limit_hit_extra_hold_days <= 0 else ordered["next3_open"]
        hit_ret = pd.to_numeric(hit_exit, errors="coerce") / pd.to_numeric(ordered["next_open"], errors="coerce") - 1.0
        ordered["policy_exit_ret"] = np.where(ordered["hit_limit_next_day"] == 1, hit_ret, miss_ret)
        ordered["risk_bad_trade"] = (ordered["policy_exit_ret"] <= -0.05).astype(int)
    return ordered.dropna(subset=["next_trade_date", "next_open"]).reset_index(drop=True)


def _add_labels_and_features(
    frame: pd.DataFrame,
    *,
    limit_hit_extra_hold_days: int = 1,
    limit_success_mode: str = "close",
) -> pd.DataFrame:
    out = frame.sort_values(["code", "trade_date"]).copy()
    grouped = out.groupby("code", sort=False)
    for source, target in [
        ("high", "next_high_actual"),
        ("low", "next_low_actual"),
        ("close", "next_close_actual"),
        ("limit_up", "next_limit_up_actual"),
    ]:
        if target not in out:
            out[target] = grouped[source].shift(-1)
    if "next2_open" not in out:
        out["next2_open"] = grouped["open"].shift(-2)
    if "next3_open" not in out:
        out["next3_open"] = grouped["open"].shift(-3)
    success_source = pd.DataFrame(
        {
            "next_high": out["next_high_actual"],
            "next_close": out["next_close_actual"],
            "next_limit_up": out["next_limit_up_actual"],
        },
        index=out.index,
    )
    out["hit_limit_next_day"] = _limit_success_mask(success_source, "next_", limit_success_mode).astype(int)
    out["next_open_ret"] = pd.to_numeric(out["next_open"], errors="coerce") / pd.to_numeric(out["close"], errors="coerce") - 1.0
    out["next_close_ret_from_open"] = pd.to_numeric(out["next_close_actual"], errors="coerce") / pd.to_numeric(out["next_open"], errors="coerce") - 1.0
    out["next_low_ret_from_open"] = pd.to_numeric(out["next_low_actual"], errors="coerce") / pd.to_numeric(out["next_open"], errors="coerce") - 1.0
    out["risk_bad_next_day"] = ((out["hit_limit_next_day"] == 0) & (out["next_low_ret_from_open"] <= -0.04)).astype(int)
    miss_ret = pd.to_numeric(out["next2_open"], errors="coerce") / pd.to_numeric(out["next_open"], errors="coerce") - 1.0
    hit_exit = out["next2_open"] if limit_hit_extra_hold_days <= 0 else out["next3_open"]
    hit_ret = pd.to_numeric(hit_exit, errors="coerce") / pd.to_numeric(out["next_open"], errors="coerce") - 1.0
    out["policy_exit_ret"] = np.where(out["hit_limit_next_day"] == 1, hit_ret, miss_ret)
    out["risk_bad_trade"] = (out["policy_exit_ret"] <= -0.05).astype(int)
    out["ret_1"] = pd.to_numeric(out["close"], errors="coerce") / pd.to_numeric(out["prev_close"], errors="coerce") - 1.0
    out["intraday_ret"] = pd.to_numeric(out["close"], errors="coerce") / pd.to_numeric(out["open"], errors="coerce") - 1.0
    out["range_pct"] = pd.to_numeric(out["high"], errors="coerce") / pd.to_numeric(out["low"], errors="coerce") - 1.0
    out["close_to_limit_up"] = pd.to_numeric(out["limit_up"], errors="coerce") / pd.to_numeric(out["close"], errors="coerce") - 1.0
    out["amount_log"] = np.log1p(pd.to_numeric(out["amount"], errors="coerce").clip(lower=0))
    out["adv20_log"] = np.log1p(pd.to_numeric(out["adv20_amount"], errors="coerce").clip(lower=0))
    out["is_limit_20pct"] = out["limit_band"].astype(str).str.contains("20").astype(float)
    return out.dropna(subset=["next_trade_date", "next_open", "next_high_actual", "next_limit_up_actual", "policy_exit_ret"]).reset_index(drop=True)


def walkforward_predict(dataset: pd.DataFrame, config: LimitHitResearchConfig) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    feature_cols = _feature_columns(dataset)
    predictions = []
    model_rows = []
    for year in range(config.start_year, config.end_year + 1):
        train = dataset[dataset["trade_date"].astype(str) < f"{year}-01-01"].copy()
        test = dataset[dataset["trade_date"].astype(str).str[:4].astype(int) == year].copy()
        if test.empty or train.empty:
            continue
        train_sample = _balanced_train_sample(train, config)
        alpha_model = _fit_classifier(train_sample[feature_cols], train_sample["hit_limit_next_day"], config.alpha_rounds, config.random_state + year)
        risk_label = "risk_bad_trade" if "risk_bad_trade" in train_sample else "risk_bad_next_day"
        risk_model = _fit_classifier(train_sample[feature_cols], train_sample[risk_label], config.risk_rounds, config.random_state + year + 1000)
        pred = test[["trade_date", "code", "next_trade_date", "open", "high", "low", "close", "limit_up", "limit_down", "next_open", "next_limit_up", "can_buy_next_open", "can_sell_next_open", "adv20_amount", "ret_1", "hit_limit_next_day", "risk_bad_next_day", "risk_bad_trade", "policy_exit_ret"]].copy()
        pred["p_limit_hit"] = _predict_proba(alpha_model, test[feature_cols])
        pred["risk_prob"] = _predict_proba(risk_model, test[feature_cols])
        pred["score"] = pred["p_limit_hit"] - config.risk_weight * pred["risk_prob"]
        pred["fold_year"] = year
        predictions.append(pred)
        model_rows.append(
            {
                "run_id": RUN_ID,
                "fold_year": year,
                "train_rows": int(len(train_sample)),
                "test_rows": int(len(test)),
                "positive_rate": float(train_sample["hit_limit_next_day"].mean()),
                "risk_rate": float(train_sample[risk_label].mean()),
                "risk_label": risk_label,
                "feature_count": len(feature_cols),
                "model": type(alpha_model).__name__,
            }
        )
    return pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame(), model_rows


def run_limit_hit_backtest(
    predictions: pd.DataFrame,
    bars: pd.DataFrame,
    *,
    initial_cash: float = 1_000_000.0,
    max_positions: int = 5,
    max_position_weight: float = 0.20,
    min_probability: float = 0.35,
    max_risk_prob: float = 0.40,
    risk_weight: float = 0.35,
    slippage_bps: float = 10.0,
    commission_bps: float = 3.0,
    stamp_duty_bps: float = 5.0,
    max_drawdown_stop: float = -0.30,
    drawdown_reduce_at: float = -0.20,
    reduced_max_positions: int = 2,
    limit_hit_extra_hold_days: int = 1,
    limit_success_mode: str = "touch",
    entry_min_ret: float | None = None,
    market_exposure_by_date: dict[str, float] | None = None,
) -> dict[str, object]:
    bars_by_code_date = {(str(r.code), str(r.trade_date)): r for r in bars.sort_values(["code", "trade_date"]).itertuples(index=False)}
    prediction_dates = {str(x) for x in predictions["trade_date"].dropna().unique()}
    if "next_trade_date" in predictions:
        prediction_dates |= {str(x) for x in predictions["next_trade_date"].dropna().unique()}
    trading_dates = sorted({str(x) for x in bars["trade_date"].dropna().unique()} | prediction_dates)
    if prediction_dates:
        first_prediction_date = min(prediction_dates)
        trading_dates = [date for date in trading_dates if date >= first_prediction_date]
    next_dates = {date: trading_dates[i + 1] for i, date in enumerate(trading_dates[:-1])}
    preds_by_date = {str(k): v.copy() for k, v in predictions.groupby("trade_date", sort=False)}
    cash = float(initial_cash)
    positions: dict[str, dict[str, object]] = {}
    orders: list[dict[str, object]] = []
    nav_rows: list[dict[str, object]] = []
    peak = cash

    for date in trading_dates:
        for code, pos in list(positions.items()):
            entry_date = str(pos["entry_date"])
            first_exit_date = next_dates.get(entry_date)
            delayed_exit_date = first_exit_date
            for _ in range(max(int(limit_hit_extra_hold_days), 0)):
                delayed_exit_date = next_dates.get(delayed_exit_date) if delayed_exit_date else None
            entry_bar = bars_by_code_date.get((code, entry_date))
            hit = bool(entry_bar is not None and _bar_limit_success(entry_bar, limit_success_mode))
            exit_date = delayed_exit_date if hit else first_exit_date
            if exit_date is not None and date >= exit_date:
                bar = bars_by_code_date.get((code, date))
                if bar is None or bool(getattr(bar, "is_paused", False)):
                    continue
                sell_px = float(bar.open) * (1.0 - slippage_bps / 10000.0)
                qty = float(pos["qty"])
                value = qty * sell_px
                fees = value * (commission_bps + stamp_duty_bps) / 10000.0
                cash += value - fees
                orders.append(
                    {
                        "trade_date": date,
                        "code": code,
                        "side": "sell",
                        "qty": qty,
                        "price": sell_px,
                        "value": value,
                        "fees": fees,
                        "exit_reason": "limit_hit_delay_exit" if hit else "miss_limit_exit",
                        "entry_date": entry_date,
                    }
                )
                positions.pop(code, None)

        nav = cash + sum(float(pos["qty"]) * _close_or_open(bars_by_code_date.get((code, date)), float(pos["entry_price"])) for code, pos in positions.items())
        peak = max(peak, nav)
        drawdown = nav / peak - 1.0 if peak else 0.0
        nav_rows.append({"trade_date": date, "nav": nav, "cash": cash, "positions": len(positions), "drawdown": drawdown})
        if drawdown <= max_drawdown_stop:
            continue
        day = preds_by_date.get(date)
        if day is None or day.empty:
            continue
        exposure_scalar = float((market_exposure_by_date or {}).get(date, 1.0))
        if exposure_scalar <= 0.0:
            continue
        allowed_positions = reduced_max_positions if drawdown <= drawdown_reduce_at else max_positions
        slots = max(0, int(allowed_positions) - len(positions))
        if slots <= 0:
            continue
        candidates = day.copy()
        candidates["score"] = pd.to_numeric(candidates.get("p_limit_hit"), errors="coerce") - risk_weight * pd.to_numeric(candidates.get("risk_prob"), errors="coerce").fillna(1.0)
        candidates = candidates[
            (pd.to_numeric(candidates["p_limit_hit"], errors="coerce") >= min_probability)
            & (pd.to_numeric(candidates["risk_prob"], errors="coerce").fillna(1.0) <= max_risk_prob)
            & (~candidates["code"].astype(str).isin(positions))
        ].sort_values(["score", "p_limit_hit", "code"], ascending=[False, False, True])
        if entry_min_ret is not None and "ret_1" in candidates:
            candidates = candidates[pd.to_numeric(candidates["ret_1"], errors="coerce").fillna(-999.0) >= float(entry_min_ret)]
        for row in candidates.head(slots).itertuples(index=False):
            entry_date = str(getattr(row, "next_trade_date", None) or next_dates.get(date, ""))
            if not entry_date:
                continue
            bar = bars_by_code_date.get((str(row.code), entry_date))
            if bar is None or bool(getattr(bar, "is_paused", False)):
                continue
            if float(bar.open) >= float(bar.limit_up) * 0.999:
                continue
            buy_px = float(bar.open) * (1.0 + slippage_bps / 10000.0)
            allocation = nav * min(1.0 / max(float(max_positions), 1.0), float(max_position_weight)) * min(exposure_scalar, 1.0)
            qty = np.floor((allocation / buy_px) / 100.0) * 100.0
            cost = qty * buy_px
            fee = cost * commission_bps / 10000.0
            if qty <= 0 or cost + fee > cash:
                continue
            cash -= cost + fee
            positions[str(row.code)] = {"qty": qty, "entry_date": entry_date, "entry_price": buy_px}
            orders.append(
                {
                    "trade_date": entry_date,
                    "decision_date": date,
                    "code": str(row.code),
                    "side": "buy",
                    "qty": qty,
                    "price": buy_px,
                    "value": cost,
                    "fees": fee,
                    "p_limit_hit": float(row.p_limit_hit),
                    "risk_prob": float(row.risk_prob),
                }
            )

    nav_frame = pd.DataFrame(nav_rows)
    order_columns = [
        "trade_date",
        "decision_date",
        "code",
        "side",
        "qty",
        "price",
        "value",
        "fees",
        "p_limit_hit",
        "risk_prob",
        "exit_reason",
        "entry_date",
    ]
    orders_frame = pd.DataFrame(orders, columns=order_columns)
    metrics = _portfolio_metrics(nav_frame)
    yearly = _yearly_metrics(nav_frame)
    return {"nav": nav_frame, "orders": orders_frame, "metrics": metrics, "yearly_metrics": yearly}


def _balanced_train_sample(train: pd.DataFrame, config: LimitHitResearchConfig) -> pd.DataFrame:
    pos = train[train["hit_limit_next_day"] == 1]
    neg = train[train["hit_limit_next_day"] == 0]
    if pos.empty or neg.empty:
        return train
    n_neg = min(len(neg), int(max(len(pos) * config.negative_sample_ratio, 1)))
    sampled_neg = neg.sample(n=n_neg, random_state=config.random_state)
    return pd.concat([pos, sampled_neg], ignore_index=True).sample(frac=1.0, random_state=config.random_state).reset_index(drop=True)


def _fit_classifier(x: pd.DataFrame, y: pd.Series, rounds: int, random_state: int):
    y = pd.to_numeric(y, errors="coerce").fillna(0).astype(int)
    x = x.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    if y.nunique() < 2:
        return _ConstantModel(float(y.mean() if len(y) else 0.0))
    try:
        from lightgbm import LGBMClassifier

        model = LGBMClassifier(
            n_estimators=int(rounds),
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            verbose=-1,
            n_jobs=-1,
        )
        model.fit(x, y)
        return model
    except Exception:
        from sklearn.ensemble import HistGradientBoostingClassifier

        model = HistGradientBoostingClassifier(max_iter=int(rounds), random_state=random_state, learning_rate=0.05)
        model.fit(x, y)
        return model


def _predict_proba(model, x: pd.DataFrame) -> np.ndarray:
    x = x.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(x)[:, 1], dtype=float)
    return np.asarray(model.predict(x), dtype=float)


class _ConstantModel:
    def __init__(self, probability: float) -> None:
        self.probability = float(probability)

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        p = np.full(len(x), self.probability)
        return np.vstack([1.0 - p, p]).T


def _feature_columns(frame: pd.DataFrame) -> list[str]:
    preferred = [
        "ret_1",
        "intraday_ret",
        "range_pct",
        "close_to_limit_up",
        "limit_up_pct",
        "turnover_rate",
        "amount_log",
        "adv20_log",
        "is_limit_20pct",
        "market_score",
        "sector_score",
        "self_score",
        "relative_strength_score",
        "resonance_score",
        "confidence",
        "ret_5w",
        "ret_10w",
        "ret_20w",
        "ret_60w",
        "vol_rvol_5",
        "vol_rvol_20",
        "range_rvol_5",
        "range_rvol_20",
        "price_position_20",
        "price_position_60",
        "ma_slope_20",
        "ma_slope_60",
        "sequence_strength_20",
        "sequence_strength_60",
    ]
    return [col for col in preferred if col in frame.columns]


def _limit_success_mask(frame: pd.DataFrame, prefix: str, mode: str) -> pd.Series:
    if mode == "close":
        source = pd.to_numeric(frame[f"{prefix}close"], errors="coerce")
    elif mode == "touch":
        source = pd.to_numeric(frame[f"{prefix}high"], errors="coerce")
    else:
        raise ValueError(f"unknown limit_success_mode: {mode}")
    limit_up = pd.to_numeric(frame[f"{prefix}limit_up"], errors="coerce")
    return source >= limit_up * 0.999


def _bar_limit_success(bar: object, mode: str) -> bool:
    source = getattr(bar, "close" if mode == "close" else "high")
    return float(source) >= float(getattr(bar, "limit_up")) * 0.999


def _portfolio_metrics(nav: pd.DataFrame) -> dict[str, float]:
    if nav.empty:
        return {"total_return": 0.0, "max_drawdown": 0.0, "sharpe": 0.0, "calmar": 0.0}
    values = pd.to_numeric(nav["nav"], errors="coerce")
    total_return = float(values.iloc[-1] / values.iloc[0] - 1.0) if len(values) > 1 and values.iloc[0] else 0.0
    daily = values.pct_change().dropna()
    sharpe = float(daily.mean() / daily.std(ddof=0) * np.sqrt(252)) if len(daily) and daily.std(ddof=0) else 0.0
    max_drawdown = float((values / values.cummax() - 1.0).min())
    years = max(len(values) / 252.0, 1e-9)
    annual_return = float((1.0 + total_return) ** (1.0 / years) - 1.0) if total_return > -1 else -1.0
    calmar = float(annual_return / abs(max_drawdown)) if max_drawdown < 0 else 0.0
    return {"total_return": total_return, "annual_return": annual_return, "max_drawdown": max_drawdown, "sharpe": sharpe, "calmar": calmar}


def _yearly_metrics(nav: pd.DataFrame) -> list[dict[str, float | int]]:
    rows = []
    if nav.empty:
        return rows
    work = nav.copy()
    work["year"] = work["trade_date"].astype(str).str[:4].astype(int)
    for year, group in work.groupby("year", sort=True):
        metrics = _portfolio_metrics(group)
        rows.append({"year": int(year), **metrics})
    return rows


def _close_or_open(bar: object | None, fallback: float) -> float:
    if bar is None:
        return fallback
    value = getattr(bar, "close", None)
    if value is None or pd.isna(value):
        value = getattr(bar, "open", fallback)
    return float(value)


def _init_research_db(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("create schema if not exists main")


def _duckdb_path_literal(path: Path) -> str:
    return str(path).replace("'", "''")


def _replace_table(con: duckdb.DuckDBPyConnection, table_name: str, frame: pd.DataFrame) -> None:
    con.execute(f"drop table if exists {table_name}")
    con.register("_replace_source", frame)
    con.execute(f"create table {table_name} as select * from _replace_source")
    con.unregister("_replace_source")


if __name__ == "__main__":
    main()
