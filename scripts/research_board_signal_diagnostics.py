from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


DEFAULT_PREDICTIONS = "outputs/limit_hit_research/reports/board_overnight_model_2020_2025/board_overnight_predictions.csv"
DEFAULT_OUT_DIR = "outputs/limit_hit_research/reports/board_signal_diagnostics_2020_2025"


@dataclass(frozen=True)
class SignalDiagnosticsConfig:
    top_n: int = 5
    min_segment_rows: int = 50


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default=DEFAULT_PREDICTIONS)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--min-segment-rows", type=int, default=50)
    args = parser.parse_args()
    run_signal_diagnostics(
        predictions_path=Path(args.predictions),
        out_dir=Path(args.out_dir),
        config=SignalDiagnosticsConfig(top_n=args.top_n, min_segment_rows=args.min_segment_rows),
    )


def run_signal_diagnostics(
    *,
    predictions_path: Path,
    out_dir: Path,
    config: SignalDiagnosticsConfig | None = None,
) -> dict[str, pd.DataFrame]:
    config = config or SignalDiagnosticsConfig()
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions = prepare_signal_frame(pd.read_csv(predictions_path), top_n=config.top_n)
    single = single_factor_segments(predictions, config=config)
    pairwise = pairwise_segments(predictions, config=config)
    profile = top_candidate_profile(predictions)
    single.to_csv(out_dir / "board_signal_single_factor_segments.csv", index=False)
    pairwise.to_csv(out_dir / "board_signal_pairwise_segments.csv", index=False)
    profile.to_csv(out_dir / "board_signal_top_candidate_profile.csv", index=False)
    manifest = {
        "predictions": str(predictions_path),
        "rows": int(len(predictions)),
        "config": asdict(config),
        "note": "Explains sealed-board overnight top candidates. Daily upper-bound study only; not live executable without fillability gate.",
    }
    (out_dir / "board_signal_diagnostics_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print("single factor top rows")
    print(single.head(30).to_string(index=False))
    print("top candidate profile")
    print(profile.to_string(index=False))
    print(f"wrote {out_dir}")
    return {"single": single, "pairwise": pairwise, "profile": profile}


def prepare_signal_frame(predictions: pd.DataFrame, *, top_n: int) -> pd.DataFrame:
    required = {
        "trade_date",
        "code",
        "target_ret_net",
        "target_win",
        "second_board_success",
        "ret_1",
        "turnover_rate",
        "adv20_amount",
        "limit_band_clean",
        "prev_up_ratio",
        "prev_sealed_count",
        "pred_ret",
        "pred_win_prob",
    }
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"missing prediction columns: {', '.join(missing)}")
    out = predictions.copy()
    out["trade_date"] = out["trade_date"].astype(str)
    out["code"] = out["code"].astype(str)
    for col in [
        "target_ret_net",
        "target_win",
        "ret_1",
        "turnover_rate",
        "adv20_amount",
        "prev_up_ratio",
        "prev_sealed_count",
        "pred_ret",
        "pred_win_prob",
    ]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["second_board_success"] = out["second_board_success"].astype(bool)
    out = out.dropna(subset=["target_ret_net", "pred_ret", "pred_win_prob"])
    ranked = out.sort_values(["trade_date", "pred_ret", "pred_win_prob", "code"], ascending=[True, False, False, True]).copy()
    ranked["daily_rank"] = ranked.groupby("trade_date").cumcount() + 1
    ranked["is_top_candidate"] = ranked["daily_rank"] <= int(top_n)
    return add_signal_buckets(ranked)


def add_signal_buckets(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["pred_ret_bucket"] = _qbucket(out["pred_ret"], 5, prefix="pred")
    out["pred_prob_bucket"] = _qbucket(out["pred_win_prob"], 5, prefix="prob")
    out["turnover_bucket"] = _qbucket(out["turnover_rate"], 5, prefix="turn")
    out["adv20_bucket"] = _qbucket(np.log1p(out["adv20_amount"].clip(lower=0)), 5, prefix="adv")
    out["ret1_bucket"] = _fixed_bucket(
        out["ret_1"],
        bins=[-np.inf, 0.08, 0.095, 0.105, 0.15, np.inf],
        labels=["lt8", "8_9p5", "9p5_10p5", "10p5_15", "gt15"],
    )
    out["prev_up_bucket"] = _fixed_bucket(
        out["prev_up_ratio"],
        bins=[-np.inf, 0.3, 0.4, 0.5, 0.6, 0.8, np.inf],
        labels=["lt30", "30_40", "40_50", "50_60", "60_80", "gt80"],
    )
    out["heat_bucket"] = _fixed_bucket(
        out["prev_sealed_count"],
        bins=[-np.inf, 20, 40, 60, 100, 200, np.inf],
        labels=["lt20", "20_40", "40_60", "60_100", "100_200", "gt200"],
    )
    return out


def single_factor_segments(frame: pd.DataFrame, *, config: SignalDiagnosticsConfig) -> pd.DataFrame:
    rows = []
    for universe, subset in [("all_sealed", frame), (f"top{config.top_n}", frame[frame["is_top_candidate"]])]:
        for col in [
            "pred_ret_bucket",
            "pred_prob_bucket",
            "turnover_bucket",
            "adv20_bucket",
            "ret1_bucket",
            "prev_up_bucket",
            "heat_bucket",
            "limit_band_clean",
        ]:
            rows.extend(_segment_rows(subset, [col], universe=universe, min_rows=config.min_segment_rows))
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["universe", "factor", "target_ret_net_mean"], ascending=[True, True, False]).reset_index(drop=True)


def pairwise_segments(frame: pd.DataFrame, *, config: SignalDiagnosticsConfig) -> pd.DataFrame:
    rows = []
    top = frame[frame["is_top_candidate"]].copy()
    for cols in [
        ["heat_bucket", "turnover_bucket"],
        ["heat_bucket", "prev_up_bucket"],
        ["heat_bucket", "limit_band_clean"],
        ["pred_ret_bucket", "turnover_bucket"],
        ["pred_ret_bucket", "heat_bucket"],
        ["ret1_bucket", "turnover_bucket"],
    ]:
        rows.extend(_segment_rows(top, cols, universe=f"top{config.top_n}", min_rows=config.min_segment_rows))
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["target_ret_net_mean", "rows"], ascending=[False, False]).reset_index(drop=True)


def top_candidate_profile(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for universe, subset in [("all_sealed", frame), ("top_candidate", frame[frame["is_top_candidate"]])]:
        rows.append(
            {
                "universe": universe,
                "rows": int(len(subset)),
                "target_ret_net_mean": _mean(subset["target_ret_net"]),
                "win_rate": _mean(subset["target_win"]),
                "second_board_rate": _mean(subset["second_board_success"].astype(float)),
                "pred_ret_mean": _mean(subset["pred_ret"]),
                "pred_win_prob_mean": _mean(subset["pred_win_prob"]),
                "turnover_rate_mean": _mean(subset["turnover_rate"]),
                "adv20_amount_median": float(pd.to_numeric(subset["adv20_amount"], errors="coerce").median()) if len(subset) else 0.0,
                "prev_up_ratio_mean": _mean(subset["prev_up_ratio"]),
                "prev_sealed_count_mean": _mean(subset["prev_sealed_count"]),
            }
        )
    return pd.DataFrame(rows)


def _segment_rows(frame: pd.DataFrame, group_cols: list[str], *, universe: str, min_rows: int) -> list[dict[str, object]]:
    rows = []
    if frame.empty:
        return rows
    grouped = frame.groupby(group_cols, dropna=False, sort=True)
    for keys, group in grouped:
        if len(group) < min_rows:
            continue
        if not isinstance(keys, tuple):
            keys = (keys,)
        pnl = pd.to_numeric(group["target_ret_net"], errors="coerce")
        rows.append(
            {
                "universe": universe,
                "factor": "+".join(group_cols),
                "bucket": "|".join(str(k) for k in keys),
                "rows": int(len(group)),
                "share": float(len(group) / len(frame)),
                "target_ret_net_mean": _mean(pnl),
                "target_ret_net_median": float(pnl.median()) if len(pnl) else 0.0,
                "win_rate": _mean(group["target_win"]),
                "second_board_rate": _mean(group["second_board_success"].astype(float)),
                "pred_ret_mean": _mean(group["pred_ret"]),
                "pred_win_prob_mean": _mean(group["pred_win_prob"]),
                "turnover_rate_mean": _mean(group["turnover_rate"]),
                "adv20_amount_median": float(pd.to_numeric(group["adv20_amount"], errors="coerce").median()),
                "prev_up_ratio_mean": _mean(group["prev_up_ratio"]),
                "prev_sealed_count_mean": _mean(group["prev_sealed_count"]),
            }
        )
    return rows


def _qbucket(values: pd.Series, q: int, *, prefix: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    labels = [f"{prefix}_q{i}" for i in range(1, q + 1)]
    try:
        return pd.qcut(numeric.rank(method="first"), q, labels=labels).astype(str)
    except ValueError:
        return pd.Series([f"{prefix}_unknown"] * len(values), index=values.index)


def _fixed_bucket(values: pd.Series, *, bins: list[float], labels: list[str]) -> pd.Series:
    return pd.cut(pd.to_numeric(values, errors="coerce"), bins=bins, labels=labels, include_lowest=True).astype(str)


def _mean(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float(numeric.mean()) if len(numeric) else 0.0


if __name__ == "__main__":
    main()
