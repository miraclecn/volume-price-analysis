from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


DEFAULT_SHARED_ML_DB = "outputs/ml/ml_ret5_alpha_risk_20260619.duckdb"
DEFAULT_OUT_DIR = "outputs/limit_hit_research/reports/board_taxonomy_2020_2025"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shared-ml-db", default=DEFAULT_SHARED_ML_DB)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--start-date", default="2020-01-01")
    parser.add_argument("--end-date", default="2025-12-31")
    parser.add_argument("--min-adv20-amount", type=float, default=10_000_000.0)
    parser.add_argument("--slippage-bps", type=float, default=10.0)
    parser.add_argument("--commission-bps", type=float, default=3.0)
    parser.add_argument("--stamp-duty-bps", type=float, default=5.0)
    args = parser.parse_args()
    run_board_taxonomy(
        shared_ml_db=Path(args.shared_ml_db),
        out_dir=Path(args.out_dir),
        start_date=args.start_date,
        end_date=args.end_date,
        min_adv20_amount=args.min_adv20_amount,
        slippage_bps=args.slippage_bps,
        commission_bps=args.commission_bps,
        stamp_duty_bps=args.stamp_duty_bps,
    )


def run_board_taxonomy(
    *,
    shared_ml_db: Path,
    out_dir: Path,
    start_date: str,
    end_date: str,
    min_adv20_amount: float,
    slippage_bps: float,
    commission_bps: float,
    stamp_duty_bps: float,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(shared_ml_db), read_only=True)
    try:
        frame = load_board_frame(con, start_date, end_date, min_adv20_amount)
    finally:
        con.close()
    labeled = add_board_taxonomy_labels(
        frame,
        slippage_bps=slippage_bps,
        commission_bps=commission_bps,
        stamp_duty_bps=stamp_duty_bps,
    )
    daily = daily_market_board_stats(labeled)
    labeled = labeled.merge(
        daily[["trade_date", "up_ratio", "sealed_count", "failed_count", "touch_count", "prev_up_ratio", "prev_sealed_count", "prev_failed_count"]],
        on="trade_date",
        how="left",
    )
    summaries = build_taxonomy_summaries(labeled)
    labeled_sample = labeled.sort_values(["trade_date", "code"]).head(20000)
    labeled_sample.to_csv(out_dir / "board_taxonomy_sample.csv", index=False)
    daily.to_csv(out_dir / "board_daily_market_stats.csv", index=False)
    for name, frame_out in summaries.items():
        frame_out.to_csv(out_dir / f"{name}.csv", index=False)
    manifest = {
        "shared_ml_db": str(shared_ml_db),
        "start_date": start_date,
        "end_date": end_date,
        "min_adv20_amount": min_adv20_amount,
        "slippage_bps": slippage_bps,
        "commission_bps": commission_bps,
        "stamp_duty_bps": stamp_duty_bps,
        "rows": int(len(labeled)),
        "sealed_rows": int(labeled["sealed_today"].sum()),
        "failed_board_rows": int(labeled["failed_board_today"].sum()),
        "summary_files": sorted(f"{name}.csv" for name in summaries),
    }
    (out_dir / "board_taxonomy_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    print(summaries["event_summary"].to_string(index=False))
    print(summaries["sealed_relay_segments"].head(20).to_string(index=False))


def load_board_frame(
    con: duckdb.DuckDBPyConnection,
    start_date: str,
    end_date: str,
    min_adv20_amount: float,
) -> pd.DataFrame:
    return con.execute(
        """
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
            t.next_trade_date,
            n.open as next_open,
            n.high as next_high,
            n.low as next_low,
            n.close as next_close,
            n.limit_up as next_limit_up,
            n.limit_down as next_limit_down,
            n.is_paused as next_is_paused,
            n.next_trade_date as next2_trade_date,
            n2.open as next2_open,
            n2.high as next2_high,
            n2.low as next2_low,
            n2.close as next2_close,
            n2.limit_up as next2_limit_up,
            n2.is_paused as next2_is_paused
        from ml_tradeability_daily t
        left join ml_tradeability_daily n
          on n.trade_date = t.next_trade_date
         and n.code = t.code
        left join ml_tradeability_daily n2
          on n2.trade_date = n.next_trade_date
         and n2.code = t.code
        where t.trade_date between ? and ?
          and t.close > 0
          and t.prev_close > 0
          and t.limit_up > 0
          and coalesce(t.is_st, false) = false
          and coalesce(t.is_paused, false) = false
          and coalesce(t.is_bse, false) = false
          and coalesce(t.adv20_amount, 0) >= ?
        order by t.trade_date, t.code
        """,
        [start_date, end_date, min_adv20_amount],
    ).fetchdf()


def add_board_taxonomy_labels(
    frame: pd.DataFrame,
    *,
    slippage_bps: float = 10.0,
    commission_bps: float = 3.0,
    stamp_duty_bps: float = 5.0,
) -> pd.DataFrame:
    out = frame.copy()
    out["ret_1"] = _num(out["close"]) / _num(out["prev_close"]) - 1.0
    out["intraday_ret"] = _num(out["close"]) / _num(out["open"]) - 1.0
    out["touch_board_today"] = _num(out["high"]) >= _num(out["limit_up"]) * 0.999
    out["sealed_today"] = _num(out["close"]) >= _num(out["limit_up"]) * 0.999
    out["failed_board_today"] = out["touch_board_today"] & ~out["sealed_today"]
    out["near_board_close"] = _num(out["close"]) >= _num(out["limit_up"]) * 0.97
    out["next_touch_board"] = _num(out["next_high"]) >= _num(out["next_limit_up"]) * 0.999
    out["second_board_success"] = _num(out["next_close"]) >= _num(out["next_limit_up"]) * 0.999
    out["next_failed_board"] = out["next_touch_board"] & ~out["second_board_success"]
    out["board_next_open_ret"] = _num(out["next_open"]) / _num(out["close"]) - 1.0
    out["board_next_close_ret"] = _num(out["next_close"]) / _num(out["close"]) - 1.0
    out["relay_next2_open_ret"] = _num(out["next2_open"]) / _num(out["next_open"]) - 1.0
    out["relay_next_close_ret"] = _num(out["next_close"]) / _num(out["next_open"]) - 1.0
    buy_cost = (slippage_bps + commission_bps) / 10000.0
    sell_cost = (slippage_bps + stamp_duty_bps) / 10000.0
    out["board_next_open_ret_net"] = out["board_next_open_ret"] - sell_cost
    out["relay_next2_open_ret_net"] = out["relay_next2_open_ret"] - buy_cost - sell_cost
    out["limit_band_clean"] = out["limit_band"].fillna("unknown").astype(str) if "limit_band" in out else "unknown"
    return out


def daily_market_board_stats(frame: pd.DataFrame) -> pd.DataFrame:
    daily = (
        frame.groupby("trade_date", as_index=False)
        .agg(
            stock_count=("code", "count"),
            up_ratio=("ret_1", lambda x: float((_num(x) > 0).mean())),
            avg_ret=("ret_1", "mean"),
            touch_count=("touch_board_today", "sum"),
            sealed_count=("sealed_today", "sum"),
            failed_count=("failed_board_today", "sum"),
            near_board_count=("near_board_close", "sum"),
        )
        .sort_values("trade_date")
        .reset_index(drop=True)
    )
    daily["seal_rate_among_touched"] = daily["sealed_count"] / daily["touch_count"].replace(0, np.nan)
    for column in ["up_ratio", "avg_ret", "sealed_count", "failed_count", "touch_count", "seal_rate_among_touched"]:
        daily[f"prev_{column}"] = daily[column].shift(1)
    return daily


def build_taxonomy_summaries(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    sealed = frame[frame["sealed_today"]].copy()
    failed = frame[frame["failed_board_today"]].copy()
    near = frame[frame["near_board_close"] & ~frame["sealed_today"]].copy()
    event_summary = pd.DataFrame(
        [
            _event_row("sealed_today", sealed),
            _event_row("failed_board_today", failed),
            _event_row("near_board_not_sealed", near),
        ]
    )
    sealed = _add_bins(sealed)
    failed = _add_bins(failed)
    relay_candidates = frame[(frame["ret_1"] >= 0.095) & (frame["near_board_close"])].copy()
    relay_candidates = _add_bins(relay_candidates)
    return {
        "event_summary": event_summary,
        "sealed_relay_segments": _segment_summary(
            sealed,
            ["prev_up_bin", "prev_sealed_bin", "limit_band_clean"],
            value_cols=["board_next_open_ret_net", "relay_next2_open_ret_net", "second_board_success", "next_failed_board"],
        ),
        "sealed_by_ret_turnover": _segment_summary(
            sealed,
            ["ret_1_bin", "turnover_bin"],
            value_cols=["board_next_open_ret_net", "relay_next2_open_ret_net", "second_board_success", "next_failed_board"],
        ),
        "failed_board_segments": _segment_summary(
            failed,
            ["prev_up_bin", "prev_sealed_bin", "limit_band_clean"],
            value_cols=["board_next_open_ret_net", "relay_next2_open_ret_net", "second_board_success", "next_failed_board"],
        ),
        "relay_candidate_segments": _segment_summary(
            relay_candidates,
            ["prev_up_bin", "preturn_bin", "turnover_bin"],
            value_cols=["relay_next2_open_ret_net", "second_board_success", "next_failed_board"],
        ),
    }


def _event_row(name: str, frame: pd.DataFrame) -> dict[str, object]:
    return {
        "event": name,
        "count": int(len(frame)),
        "board_next_open_ret_net_mean": _mean(frame, "board_next_open_ret_net"),
        "board_next_open_win_rate": _win_rate(frame, "board_next_open_ret_net"),
        "relay_next2_open_ret_net_mean": _mean(frame, "relay_next2_open_ret_net"),
        "relay_next2_open_win_rate": _win_rate(frame, "relay_next2_open_ret_net"),
        "second_board_success_rate": _mean(frame, "second_board_success"),
        "next_failed_board_rate": _mean(frame, "next_failed_board"),
    }


def _add_bins(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["prev_up_bin"] = pd.cut(_num(out["prev_up_ratio"]), [0, 0.30, 0.40, 0.50, 0.60, 1.0], include_lowest=True)
    out["prev_sealed_bin"] = pd.cut(_num(out["prev_sealed_count"]), [-1, 20, 50, 100, 200, 100000], include_lowest=True)
    out["ret_1_bin"] = pd.cut(_num(out["ret_1"]), [-1, 0.05, 0.095, 0.105, 0.15, 0.25, 1.0], include_lowest=True)
    out["preturn_bin"] = pd.cut(_num(out["ret_1"]), [0.095, 0.105, 0.15, 0.25, 1.0], include_lowest=True)
    out["turnover_bin"] = pd.qcut(_num(out["turnover_rate"]).rank(method="first"), 5, duplicates="drop")
    return out


def _segment_summary(frame: pd.DataFrame, group_cols: list[str], *, value_cols: list[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=group_cols + ["count"])
    rows = []
    for keys, group in frame.groupby(group_cols, observed=True, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: str(value) for col, value in zip(group_cols, keys)}
        row["count"] = int(len(group))
        for col in value_cols:
            row[f"{col}_mean"] = _mean(group, col)
            if col.endswith("_ret_net"):
                row[f"{col}_win_rate"] = _win_rate(group, col)
        rows.append(row)
    out = pd.DataFrame(rows)
    if "relay_next2_open_ret_net_mean" in out:
        return out.sort_values(["relay_next2_open_ret_net_mean", "count"], ascending=[False, False]).reset_index(drop=True)
    return out.sort_values(["count"], ascending=False).reset_index(drop=True)


def _num(values: object) -> pd.Series:
    return pd.to_numeric(values, errors="coerce")


def _mean(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame:
        return 0.0
    value = pd.to_numeric(frame[column], errors="coerce").mean()
    return float(value) if pd.notna(value) else 0.0


def _win_rate(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame:
        return 0.0
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float((values > 0).mean()) if len(values) else 0.0


if __name__ == "__main__":
    main()
