from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


DEFAULT_OVERLAY_DIR = "outputs/limit_hit_research/reports/board_overlay_profit_protect_2020_2025"
DEFAULT_VARIANT = "board_neutral_expected_scale10"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overlay-dir", default=DEFAULT_OVERLAY_DIR)
    parser.add_argument("--variant", default=DEFAULT_VARIANT)
    args = parser.parse_args()
    out = run_overlay_diagnostics(Path(args.overlay_dir), args.variant)
    print(json.dumps(out["summary"], indent=2, sort_keys=True))


def run_overlay_diagnostics(overlay_dir: Path, variant: str) -> dict[str, object]:
    nav_path = overlay_dir / f"{variant}_nav.csv"
    if not nav_path.exists():
        raise FileNotFoundError(nav_path)
    nav = pd.read_csv(nav_path)
    diagnostics = overlay_diagnostics(nav)
    yearly = yearly_contribution(nav)
    monthly = monthly_contribution(nav)
    worst_days = worst_days_table(nav)
    diagnostics.to_csv(overlay_dir / f"{variant}_diagnostics.csv", index=False)
    yearly.to_csv(overlay_dir / f"{variant}_yearly_contribution.csv", index=False)
    monthly.to_csv(overlay_dir / f"{variant}_monthly_contribution.csv", index=False)
    worst_days.to_csv(overlay_dir / f"{variant}_worst_days.csv", index=False)
    summary = diagnostics.iloc[0].to_dict()
    (overlay_dir / f"{variant}_diagnostics_summary.json").write_text(
        json.dumps(_jsonable(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return {"summary": summary, "yearly": yearly, "monthly": monthly, "worst_days": worst_days}


def overlay_diagnostics(nav: pd.DataFrame) -> pd.DataFrame:
    frame = _prepare(nav)
    corr = _corr(frame["main_return"], frame["board_return"])
    main_neg = frame[frame["main_return"] < 0]
    board_on_main_down = float(main_neg["board_return"].mean()) if len(main_neg) else 0.0
    both_down_rate = float(((frame["main_return"] < 0) & (frame["board_return"] < 0)).mean()) if len(frame) else 0.0
    board_share = _safe_div(frame["board_return"].sum(), frame["combined_return"].sum())
    return pd.DataFrame(
        [
            {
                "rows": int(len(frame)),
                "main_board_corr": corr,
                "board_mean_return_on_main_down_days": board_on_main_down,
                "both_down_rate": both_down_rate,
                "board_sum_return_share": board_share,
                "worst_combined_day_return": float(frame["combined_return"].min()) if len(frame) else 0.0,
                "worst_main_day_return": float(frame["main_return"].min()) if len(frame) else 0.0,
                "worst_board_day_return": float(frame["board_return"].min()) if len(frame) else 0.0,
            }
        ]
    )


def yearly_contribution(nav: pd.DataFrame) -> pd.DataFrame:
    frame = _prepare(nav)
    frame["year"] = frame["trade_date"].astype(str).str[:4].astype(int)
    rows = []
    for year, group in frame.groupby("year", sort=True):
        rows.append(_contribution_row(int(year), group))
    return pd.DataFrame(rows)


def monthly_contribution(nav: pd.DataFrame) -> pd.DataFrame:
    frame = _prepare(nav)
    frame["month"] = frame["trade_date"].astype(str).str[:7]
    rows = []
    for month, group in frame.groupby("month", sort=True):
        row = _contribution_row(month, group)
        rows.append(row)
    return pd.DataFrame(rows)


def worst_days_table(nav: pd.DataFrame, *, n: int = 20) -> pd.DataFrame:
    frame = _prepare(nav)
    return frame.sort_values("combined_return").head(n)[["trade_date", "main_return", "board_return", "combined_return", "drawdown"]].reset_index(drop=True)


def _contribution_row(period: int | str, group: pd.DataFrame) -> dict[str, object]:
    return {
        "period": period,
        "rows": int(len(group)),
        "main_return_sum": float(group["main_return"].sum()),
        "board_return_sum": float(group["board_return"].sum()),
        "combined_return_sum": float(group["combined_return"].sum()),
        "main_board_corr": _corr(group["main_return"], group["board_return"]),
        "board_mean_return_on_main_down_days": float(group.loc[group["main_return"] < 0, "board_return"].mean()) if (group["main_return"] < 0).any() else 0.0,
        "both_down_rate": float(((group["main_return"] < 0) & (group["board_return"] < 0)).mean()) if len(group) else 0.0,
    }


def _prepare(nav: pd.DataFrame) -> pd.DataFrame:
    required = {"trade_date", "main_return", "board_return", "combined_return", "drawdown"}
    missing = sorted(required - set(nav.columns))
    if missing:
        raise ValueError(f"missing overlay nav columns: {', '.join(missing)}")
    out = nav.copy()
    out["trade_date"] = out["trade_date"].astype(str)
    for col in ["main_return", "board_return", "combined_return", "drawdown"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    return out


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def _corr(left: pd.Series, right: pd.Series) -> float:
    if len(left) <= 1:
        return 0.0
    if left.std(ddof=0) == 0 or right.std(ddof=0) == 0:
        return 0.0
    value = left.corr(right)
    return 0.0 if pd.isna(value) else float(value)


def _jsonable(row: dict[str, object]) -> dict[str, object]:
    out = {}
    for key, value in row.items():
        if hasattr(value, "item"):
            value = value.item()
        out[key] = value
    return out


if __name__ == "__main__":
    main()
