from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.research_limit_hit_strategy import load_backtest_bars, run_limit_hit_backtest


DEFAULT_SOURCE_DB = "outputs/limit_hit_research/limit_hit_research_close.duckdb"
DEFAULT_SHARED_ML_DB = "outputs/ml/ml_ret5_alpha_risk_20260619.duckdb"
DEFAULT_OUT_DIR = "outputs/limit_hit_research/reports/limit_hit_regime_sweep_2020_2025"


@dataclass(frozen=True)
class RegimeVariant:
    name: str
    max_positions: int
    max_position_weight: float
    slippage_bps: float = 10.0
    min_probability: float = 0.35
    max_risk_prob: float = 0.10
    risk_weight: float = 0.35
    entry_min_ret: float = 0.10
    market_rule: str = "none"
    zero_below: float | None = None
    half_below: float | None = None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-db", default=DEFAULT_SOURCE_DB)
    parser.add_argument("--shared-ml-db", default=DEFAULT_SHARED_ML_DB)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    run_sweep(Path(args.source_db), Path(args.shared_ml_db), Path(args.out_dir))


def run_sweep(source_db: Path, shared_ml_db: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(source_db), read_only=True)
    try:
        con.execute(f"attach '{_duckdb_path_literal(shared_ml_db)}' as shared_ml (read_only)")
        predictions = _load_enriched_predictions(con)
        bars = load_backtest_bars(con, predictions)
        market_state = _load_market_state(con, predictions)
        variants = _variants()
        metrics_rows = []
        yearly_rows = []
        attribution_rows = []
        for variant in variants:
            exposure = _market_exposure_by_date(market_state, variant)
            result = run_limit_hit_backtest(
                predictions,
                bars,
                initial_cash=1_000_000.0,
                max_positions=variant.max_positions,
                max_position_weight=variant.max_position_weight,
                min_probability=variant.min_probability,
                max_risk_prob=variant.max_risk_prob,
                risk_weight=variant.risk_weight,
                slippage_bps=variant.slippage_bps,
                commission_bps=3.0,
                stamp_duty_bps=5.0,
                max_drawdown_stop=-0.30,
                drawdown_reduce_at=-0.20,
                reduced_max_positions=2,
                limit_hit_extra_hold_days=1,
                limit_success_mode="close",
                entry_min_ret=variant.entry_min_ret,
                market_exposure_by_date=exposure,
            )
            metrics_rows.append({"variant": variant.name, **asdict(variant), **result["metrics"], "buy_count": int((result["orders"]["side"] == "buy").sum())})
            for row in result["yearly_metrics"]:
                yearly_rows.append({"variant": variant.name, **row})
            attribution_rows.extend(_trade_attribution(variant.name, result["orders"], predictions, market_state))
            result["nav"].to_csv(out_dir / f"{variant.name}_nav.csv", index=False)
            result["orders"].to_csv(out_dir / f"{variant.name}_orders.csv", index=False)
        metrics = pd.DataFrame(metrics_rows).sort_values(["annual_return", "max_drawdown"], ascending=[False, False])
        yearly = pd.DataFrame(yearly_rows)
        attribution = pd.DataFrame(attribution_rows)
        metrics.to_csv(out_dir / "regime_sweep_metrics.csv", index=False)
        yearly.to_csv(out_dir / "regime_sweep_yearly_metrics.csv", index=False)
        attribution.to_csv(out_dir / "regime_sweep_trade_attribution.csv", index=False)
        summary = {
            "source_db": str(source_db),
            "shared_ml_db": str(shared_ml_db),
            "variants": [asdict(v) for v in variants],
            "best_by_annual_return": metrics.head(10).to_dict(orient="records"),
        }
        (out_dir / "regime_sweep_manifest.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        print(metrics[["variant", "total_return", "annual_return", "max_drawdown", "sharpe", "calmar", "buy_count"]].to_string(index=False))
        print(f"wrote {out_dir}")
    finally:
        con.close()


def _load_enriched_predictions(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return con.execute(
        """
        select
            p.*,
            l.limit_band,
            l.limit_up_pct,
            l.market_score,
            l.sector_score,
            l.turnover_rate,
            l.amount,
            l.close_to_limit_up,
            l.intraday_ret,
            l.next_close_ret_from_open,
            l.next_low_ret_from_open
        from lh_predictions_daily p
        left join lh_labels_daily l
          on l.trade_date = p.trade_date
         and l.code = p.code
        where p.fold_year between 2020 and 2025
        order by p.trade_date, p.code
        """
    ).fetchdf()


def _load_market_state(con: duckdb.DuckDBPyConnection, predictions: pd.DataFrame) -> dict[str, dict[str, float]]:
    start = str(predictions["trade_date"].min())
    end = str(predictions["trade_date"].max())
    daily = con.execute(
        """
        select
            trade_date,
            avg(case when close > prev_close then 1.0 else 0.0 end) as up_ratio,
            avg(close / nullif(prev_close, 0) - 1.0) as avg_ret
        from shared_ml.ml_tradeability_daily
        where trade_date between ? and ?
          and close > 0
          and prev_close > 0
          and coalesce(is_st, false) = false
          and coalesce(is_paused, false) = false
          and coalesce(is_bse, false) = false
          and coalesce(adv20_amount, 0) >= 10000000
        group by trade_date
        order by trade_date
        """,
        [start, end],
    ).fetchdf()
    market_scores = (
        predictions[["trade_date", "market_score"]]
        .dropna()
        .groupby("trade_date", as_index=False)["market_score"]
        .median()
        .rename(columns={"market_score": "candidate_market_score"})
    )
    daily = daily.merge(market_scores, on="trade_date", how="left")
    daily["prev_up_ratio"] = daily["up_ratio"].shift(1)
    daily["prev_avg_ret"] = daily["avg_ret"].shift(1)
    daily["prev_candidate_market_score"] = daily["candidate_market_score"].shift(1)
    return {
        str(row.trade_date): {
            "up_ratio": _float(row.up_ratio, 1.0),
            "avg_ret": _float(row.avg_ret, 0.0),
            "prev_up_ratio": _float(row.prev_up_ratio, 1.0),
            "prev_avg_ret": _float(row.prev_avg_ret, 0.0),
            "prev_candidate_market_score": _float(row.prev_candidate_market_score, 50.0),
        }
        for row in daily.itertuples(index=False)
    }


def _variants() -> list[RegimeVariant]:
    variants = [
        RegimeVariant(name="base_pos1_cap069", max_positions=1, max_position_weight=0.69),
        RegimeVariant(name="base_pos1_cap062", max_positions=1, max_position_weight=0.62),
        RegimeVariant(name="broad_p030_r015_pos2_cap035", max_positions=2, max_position_weight=0.35, min_probability=0.30, max_risk_prob=0.15),
        RegimeVariant(name="broad_p025_r015_pos2_cap030", max_positions=2, max_position_weight=0.30, min_probability=0.25, max_risk_prob=0.15),
        RegimeVariant(name="broad_p020_r020_pos3_cap025", max_positions=3, max_position_weight=0.25, min_probability=0.20, max_risk_prob=0.20),
        RegimeVariant(name="broad_p030_r020_pos2_cap035", max_positions=2, max_position_weight=0.35, min_probability=0.30, max_risk_prob=0.20),
    ]
    for cap in [0.62, 0.69]:
        variants.extend(
            [
                RegimeVariant(name=f"prev_up35_45_pos1_cap{int(cap * 100):02d}", max_positions=1, max_position_weight=cap, market_rule="prev_up", zero_below=0.35, half_below=0.45),
                RegimeVariant(name=f"prev_up375_475_pos1_cap{int(cap * 100):02d}", max_positions=1, max_position_weight=cap, market_rule="prev_up", zero_below=0.375, half_below=0.475),
                RegimeVariant(name=f"prev_up40_50_pos1_cap{int(cap * 100):02d}", max_positions=1, max_position_weight=cap, market_rule="prev_up", zero_below=0.40, half_below=0.50),
                RegimeVariant(name=f"mkt_score40_50_pos1_cap{int(cap * 100):02d}", max_positions=1, max_position_weight=cap, market_rule="market_score", zero_below=40.0, half_below=50.0),
                RegimeVariant(name=f"mkt_score45_55_pos1_cap{int(cap * 100):02d}", max_positions=1, max_position_weight=cap, market_rule="market_score", zero_below=45.0, half_below=55.0),
            ]
        )
    for zero, half in [(40.0, 50.0), (45.0, 55.0)]:
        variants.extend(
            [
                RegimeVariant(name=f"broad_p030_r015_pos2_cap035_mkt{int(zero)}_{int(half)}", max_positions=2, max_position_weight=0.35, min_probability=0.30, max_risk_prob=0.15, market_rule="market_score", zero_below=zero, half_below=half),
                RegimeVariant(name=f"broad_p025_r015_pos2_cap030_mkt{int(zero)}_{int(half)}", max_positions=2, max_position_weight=0.30, min_probability=0.25, max_risk_prob=0.15, market_rule="market_score", zero_below=zero, half_below=half),
            ]
        )
    return variants


def _market_exposure_by_date(market_state: dict[str, dict[str, float]], variant: RegimeVariant) -> dict[str, float]:
    if variant.market_rule == "none":
        return {}
    out = {}
    for date, state in market_state.items():
        if variant.market_rule == "prev_up":
            value = float(state["prev_up_ratio"])
        elif variant.market_rule == "market_score":
            value = float(state["prev_candidate_market_score"])
        else:
            value = 1.0
        scalar = 1.0
        if variant.zero_below is not None and value < float(variant.zero_below):
            scalar = 0.0
        elif variant.half_below is not None and value < float(variant.half_below):
            scalar = 0.5
        out[date] = scalar
    return out


def _trade_attribution(
    variant: str,
    orders: pd.DataFrame,
    predictions: pd.DataFrame,
    market_state: dict[str, dict[str, float]],
) -> list[dict[str, object]]:
    if orders.empty:
        return []
    buys = orders[orders["side"] == "buy"].copy()
    sells = orders[orders["side"] == "sell"].copy()
    pred_index = predictions.set_index(["trade_date", "code"], drop=False)
    rows = []
    for buy in buys.itertuples(index=False):
        sell = sells[(sells["code"].astype(str) == str(buy.code)) & (sells["entry_date"].astype(str) == str(buy.trade_date))]
        if sell.empty:
            continue
        sell_row = sell.iloc[0]
        pred = pred_index.loc[(str(buy.decision_date), str(buy.code))] if (str(buy.decision_date), str(buy.code)) in pred_index.index else pd.Series(dtype=object)
        pnl = float(sell_row["value"]) - float(buy.value) - float(buy.fees or 0.0) - float(sell_row["fees"] or 0.0)
        state = market_state.get(str(buy.decision_date), {})
        rows.append(
            {
                "variant": variant,
                "decision_date": buy.decision_date,
                "entry_date": buy.trade_date,
                "code": buy.code,
                "pnl": pnl,
                "return_on_cost": pnl / float(buy.value) if float(buy.value) else 0.0,
                "is_win": pnl > 0,
                "exit_reason": sell_row["exit_reason"],
                "p_limit_hit": float(buy.p_limit_hit),
                "risk_prob": float(buy.risk_prob),
                "hit_limit_next_day": _float(pred.get("hit_limit_next_day"), 0.0),
                "ret_1": _float(pred.get("ret_1"), 0.0),
                "limit_band": pred.get("limit_band"),
                "market_score": _float(pred.get("market_score"), 0.0),
                "sector_score": _float(pred.get("sector_score"), 0.0),
                "prev_up_ratio": _float(state.get("prev_up_ratio"), 1.0),
                "prev_avg_ret": _float(state.get("prev_avg_ret"), 0.0),
            }
        )
    return rows


def _float(value: object, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _duckdb_path_literal(path: Path) -> str:
    return str(path).replace("'", "''")


if __name__ == "__main__":
    main()
