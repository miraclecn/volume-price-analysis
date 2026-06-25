from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.research_board_overnight_model import _portfolio_metrics, _yearly_metrics


DEFAULT_MAIN_NAV = "outputs/ml/reports/profit_protect_continuous_wf_2020_2025_20260622/continuous_nav.csv"
DEFAULT_OUT_DIR = "outputs/limit_hit_research/reports/board_overlay_profit_protect_2020_2025"
DEFAULT_BOARD_NAVS = {
    "board_neutral_expected": "outputs/limit_hit_research/reports/board_fill_aware_selection_2020_2025/neutral_expected_pred_ret_nav.csv",
    "board_neutral_alpha": "outputs/limit_hit_research/reports/board_fill_aware_selection_2020_2025/neutral_alpha_top_nav.csv",
    "board_conservative_expected": "outputs/limit_hit_research/reports/board_fill_aware_selection_2020_2025/conservative_expected_pred_ret_nav.csv",
    "board_severe_expected": "outputs/limit_hit_research/reports/board_fill_aware_selection_2020_2025/severe_expected_pred_ret_nav.csv",
}


@dataclass(frozen=True)
class OverlayVariant:
    name: str
    board_nav: str
    board_scale: float = 1.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-nav", default=DEFAULT_MAIN_NAV)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    run_overlay_sweep(Path(args.main_nav), Path(args.out_dir))


def run_overlay_sweep(main_nav_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    main_nav = load_nav_returns(main_nav_path, date_col="sim_date", nav_col="nav", first_return_from_initial=False)
    variants = _variants()
    metrics_rows = []
    yearly_rows = []
    for variant in variants:
        board_nav = load_nav_returns(Path(variant.board_nav), date_col="trade_date", nav_col="nav", first_return_from_initial=True)
        combined = combine_returns(main_nav, board_nav, board_scale=variant.board_scale)
        combined.to_csv(out_dir / f"{variant.name}_nav.csv", index=False)
        metrics_rows.append({"variant": variant.name, **asdict(variant), **_portfolio_metrics(combined[["trade_date", "nav"]])})
        for row in _yearly_metrics(combined[["trade_date", "nav"]]):
            yearly_rows.append({"variant": variant.name, **row})
    metrics = pd.DataFrame(metrics_rows).sort_values(["annual_return", "max_drawdown"], ascending=[False, False])
    yearly = pd.DataFrame(yearly_rows)
    metrics.to_csv(out_dir / "board_overlay_metrics.csv", index=False)
    yearly.to_csv(out_dir / "board_overlay_yearly_metrics.csv", index=False)
    manifest = {
        "main_nav": str(main_nav_path),
        "variants": [asdict(v) for v in variants],
        "note": "Research overlay only. Board leg is paper/proxy and must not be connected to live until candidate gate passes.",
    }
    (out_dir / "board_overlay_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(metrics[["variant", "total_return", "annual_return", "max_drawdown", "sharpe", "calmar"]].to_string(index=False))
    print(f"wrote {out_dir}")


def load_nav_returns(
    path: Path,
    *,
    date_col: str,
    nav_col: str,
    first_return_from_initial: bool,
    initial_nav: float = 1_000_000.0,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    if date_col not in frame.columns or nav_col not in frame.columns:
        raise ValueError(f"{path} must contain {date_col} and {nav_col}")
    out = frame[[date_col, nav_col]].copy().rename(columns={date_col: "trade_date", nav_col: "nav"})
    out["trade_date"] = out["trade_date"].astype(str)
    out["nav"] = pd.to_numeric(out["nav"], errors="coerce")
    out = out.dropna(subset=["nav"]).sort_values("trade_date").reset_index(drop=True)
    returns = out["nav"].pct_change()
    if first_return_from_initial and len(out):
        returns.iloc[0] = float(out["nav"].iloc[0]) / float(initial_nav) - 1.0
    else:
        returns.iloc[0] = 0.0 if len(out) else np.nan
    out["daily_return"] = returns.fillna(0.0)
    return out[["trade_date", "nav", "daily_return"]]


def combine_returns(
    main_returns: pd.DataFrame,
    board_returns: pd.DataFrame,
    *,
    board_scale: float,
    initial_nav: float = 1_000_000.0,
) -> pd.DataFrame:
    main = main_returns[["trade_date", "daily_return"]].rename(columns={"daily_return": "main_return"}).copy()
    board = board_returns[["trade_date", "daily_return"]].rename(columns={"daily_return": "board_return"}).copy()
    merged = main.merge(board, on="trade_date", how="left")
    merged["board_return"] = pd.to_numeric(merged["board_return"], errors="coerce").fillna(0.0)
    merged["main_return"] = pd.to_numeric(merged["main_return"], errors="coerce").fillna(0.0)
    merged["combined_return"] = merged["main_return"] + merged["board_return"] * float(board_scale)
    nav = float(initial_nav)
    peak = nav
    rows = []
    for row in merged.itertuples(index=False):
        nav *= 1.0 + float(row.combined_return)
        peak = max(peak, nav)
        rows.append(
            {
                "trade_date": row.trade_date,
                "nav": nav,
                "drawdown": nav / peak - 1.0,
                "main_return": float(row.main_return),
                "board_return": float(row.board_return),
                "combined_return": float(row.combined_return),
            }
        )
    return pd.DataFrame(rows)


def _variants() -> list[OverlayVariant]:
    variants = [
        OverlayVariant("main_only", DEFAULT_BOARD_NAVS["board_neutral_expected"], board_scale=0.0),
    ]
    for label, path in DEFAULT_BOARD_NAVS.items():
        variants.append(OverlayVariant(f"{label}_scale05", path, board_scale=0.5))
        variants.append(OverlayVariant(f"{label}_scale10", path, board_scale=1.0))
    return variants


if __name__ == "__main__":
    main()
