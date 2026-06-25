from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.research_board_taxonomy import add_board_taxonomy_labels, daily_market_board_stats, load_board_frame


DEFAULT_SHARED_ML_DB = "outputs/ml/ml_ret5_alpha_risk_20260619.duckdb"
DEFAULT_OUT_DIR = "outputs/limit_hit_research/reports/board_overnight_model_2020_2025"


@dataclass(frozen=True)
class OvernightVariant:
    name: str
    max_positions: int
    max_name_weight: float
    max_total_exposure: float
    min_pred_return: float | None = None
    min_pred_win_prob: float | None = None
    market_rule: str = "none"
    zero_below: float | None = None
    half_below: float | None = None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shared-ml-db", default=DEFAULT_SHARED_ML_DB)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--start-date", default="2020-01-01")
    parser.add_argument("--end-date", default="2025-12-31")
    parser.add_argument("--train-start", default="2015-01-05")
    parser.add_argument("--min-adv20-amount", type=float, default=10_000_000.0)
    parser.add_argument("--slippage-bps", type=float, default=10.0)
    parser.add_argument("--commission-bps", type=float, default=3.0)
    parser.add_argument("--stamp-duty-bps", type=float, default=5.0)
    args = parser.parse_args()
    run_board_overnight_model(
        shared_ml_db=Path(args.shared_ml_db),
        out_dir=Path(args.out_dir),
        train_start=args.train_start,
        start_date=args.start_date,
        end_date=args.end_date,
        min_adv20_amount=args.min_adv20_amount,
        slippage_bps=args.slippage_bps,
        commission_bps=args.commission_bps,
        stamp_duty_bps=args.stamp_duty_bps,
    )


def run_board_overnight_model(
    *,
    shared_ml_db: Path,
    out_dir: Path,
    train_start: str,
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
        raw = load_board_frame(con, train_start, end_date, min_adv20_amount)
    finally:
        con.close()
    data = prepare_overnight_dataset(
        raw,
        slippage_bps=slippage_bps,
        commission_bps=commission_bps,
        stamp_duty_bps=stamp_duty_bps,
    )
    predictions, folds = walkforward_predict_overnight(data, start_date=start_date, end_date=end_date)
    variants = _variants()
    metrics_rows = []
    yearly_rows = []
    for variant in variants:
        result = run_overnight_backtest(predictions, variant)
        result["nav"].to_csv(out_dir / f"{variant.name}_nav.csv", index=False)
        result["orders"].to_csv(out_dir / f"{variant.name}_orders.csv", index=False)
        metrics_rows.append({"variant": variant.name, **asdict(variant), **result["metrics"], "buy_count": int(len(result["orders"]))})
        for row in result["yearly_metrics"]:
            yearly_rows.append({"variant": variant.name, **row})
    metrics = pd.DataFrame(metrics_rows).sort_values(["annual_return", "max_drawdown"], ascending=[False, False])
    yearly = pd.DataFrame(yearly_rows)
    diagnostics = selection_diagnostics(predictions)
    deciles = prediction_deciles(predictions)
    predictions.to_csv(out_dir / "board_overnight_predictions.csv", index=False)
    pd.DataFrame(folds).to_csv(out_dir / "board_overnight_folds.csv", index=False)
    metrics.to_csv(out_dir / "board_overnight_metrics.csv", index=False)
    yearly.to_csv(out_dir / "board_overnight_yearly_metrics.csv", index=False)
    diagnostics.to_csv(out_dir / "board_overnight_selection_diagnostics.csv", index=False)
    deciles.to_csv(out_dir / "board_overnight_prediction_deciles.csv", index=False)
    manifest = {
        "shared_ml_db": str(shared_ml_db),
        "train_start": train_start,
        "start_date": start_date,
        "end_date": end_date,
        "min_adv20_amount": min_adv20_amount,
        "slippage_bps": slippage_bps,
        "commission_bps": commission_bps,
        "stamp_duty_bps": stamp_duty_bps,
        "rows": int(len(data)),
        "prediction_rows": int(len(predictions)),
        "variants": [asdict(v) for v in variants],
    }
    (out_dir / "board_overnight_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(metrics[["variant", "total_return", "annual_return", "max_drawdown", "sharpe", "calmar", "buy_count"]].to_string(index=False))
    print(diagnostics.to_string(index=False))
    print(f"wrote {out_dir}")


def prepare_overnight_dataset(
    raw: pd.DataFrame,
    *,
    slippage_bps: float,
    commission_bps: float,
    stamp_duty_bps: float,
) -> pd.DataFrame:
    labeled = add_board_taxonomy_labels(raw, slippage_bps=slippage_bps, commission_bps=commission_bps, stamp_duty_bps=stamp_duty_bps)
    daily = daily_market_board_stats(labeled)
    merged = labeled.merge(
        daily[
            [
                "trade_date",
                "prev_up_ratio",
                "prev_avg_ret",
                "prev_sealed_count",
                "prev_failed_count",
                "prev_touch_count",
                "prev_seal_rate_among_touched",
            ]
        ],
        on="trade_date",
        how="left",
    )
    out = merged[merged["sealed_today"]].copy()
    total_cost = (commission_bps + slippage_bps + stamp_duty_bps) / 10000.0
    out["target_ret_net"] = out["board_next_open_ret"] - total_cost
    out["target_win"] = (out["target_ret_net"] > 0).astype(int)
    out["amount_log"] = np.log1p(pd.to_numeric(out["amount"], errors="coerce").clip(lower=0))
    out["adv20_log"] = np.log1p(pd.to_numeric(out["adv20_amount"], errors="coerce").clip(lower=0))
    out["is_limit_10pct"] = out["limit_band_clean"].eq("limit_10pct").astype(float)
    out["is_limit_20pct"] = out["limit_band_clean"].eq("limit_20pct").astype(float)
    out["is_limit_unknown"] = out["limit_band_clean"].eq("unknown").astype(float)
    return out.dropna(subset=["next_open", "target_ret_net"]).reset_index(drop=True)


def walkforward_predict_overnight(data: pd.DataFrame, *, start_date: str, end_date: str) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    feature_cols = _feature_columns(data)
    start_year = int(start_date[:4])
    end_year = int(end_date[:4])
    predictions = []
    folds = []
    for year in range(start_year, end_year + 1):
        train = data[data["trade_date"].astype(str) < f"{year}-01-01"].copy()
        test = data[data["trade_date"].astype(str).str[:4].astype(int) == year].copy()
        if train.empty or test.empty:
            continue
        reg = _fit_regressor(train[feature_cols], train["target_ret_net"], random_state=20260622 + year)
        clf = _fit_classifier(train[feature_cols], train["target_win"], random_state=20261622 + year)
        pred = test[
            [
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
            ]
        ].copy()
        pred["pred_ret"] = _predict_regression(reg, test[feature_cols])
        pred["pred_win_prob"] = _predict_probability(clf, test[feature_cols])
        pred["fold_year"] = year
        predictions.append(pred)
        folds.append(
            {
                "fold_year": year,
                "train_rows": int(len(train)),
                "test_rows": int(len(test)),
                "feature_count": int(len(feature_cols)),
                "train_ret_mean": float(train["target_ret_net"].mean()),
                "test_ret_mean": float(test["target_ret_net"].mean()),
                "train_win_rate": float(train["target_win"].mean()),
                "test_win_rate": float(test["target_win"].mean()),
            }
        )
    return pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame(), folds


def run_overnight_backtest(predictions: pd.DataFrame, variant: OvernightVariant, *, initial_nav: float = 1_000_000.0) -> dict[str, pd.DataFrame | dict[str, float] | list[dict[str, float | int]]]:
    nav = float(initial_nav)
    nav_rows = []
    order_rows = []
    peak = nav
    for date, day in predictions.groupby("trade_date", sort=True):
        candidates = day.copy()
        if variant.min_pred_return is not None:
            candidates = candidates[pd.to_numeric(candidates["pred_ret"], errors="coerce") >= float(variant.min_pred_return)]
        if variant.min_pred_win_prob is not None:
            candidates = candidates[pd.to_numeric(candidates["pred_win_prob"], errors="coerce") >= float(variant.min_pred_win_prob)]
        exposure = _market_exposure(variant, candidates)
        if exposure > 0 and not candidates.empty:
            selected = candidates.sort_values(["pred_ret", "pred_win_prob", "code"], ascending=[False, False, True]).head(variant.max_positions)
            total_weight = min(float(variant.max_total_exposure), float(variant.max_name_weight) * len(selected)) * exposure
            weight = total_weight / len(selected) if len(selected) else 0.0
            day_ret = float((pd.to_numeric(selected["target_ret_net"], errors="coerce").fillna(0.0) * weight).sum())
            start_nav = nav
            nav *= 1.0 + day_ret
            for row in selected.itertuples(index=False):
                order_rows.append(
                    {
                        "trade_date": date,
                        "code": row.code,
                        "weight": weight,
                        "start_nav": start_nav,
                        "pnl": start_nav * weight * float(row.target_ret_net),
                        "target_ret_net": float(row.target_ret_net),
                        "target_win": int(row.target_win),
                        "second_board_success": bool(row.second_board_success),
                        "pred_ret": float(row.pred_ret),
                        "pred_win_prob": float(row.pred_win_prob),
                        "prev_up_ratio": float(row.prev_up_ratio) if pd.notna(row.prev_up_ratio) else np.nan,
                        "prev_sealed_count": float(row.prev_sealed_count) if pd.notna(row.prev_sealed_count) else np.nan,
                    }
                )
        peak = max(peak, nav)
        nav_rows.append({"trade_date": date, "nav": nav, "drawdown": nav / peak - 1.0})
    nav_frame = pd.DataFrame(nav_rows)
    orders = pd.DataFrame(order_rows)
    return {"nav": nav_frame, "orders": orders, "metrics": _portfolio_metrics(nav_frame), "yearly_metrics": _yearly_metrics(nav_frame)}


def selection_diagnostics(predictions: pd.DataFrame, *, total_exposure: float = 0.05, max_positions: int = 5) -> pd.DataFrame:
    rows = []
    for mode in ["top_pred", "random_fixed", "all_equal", "code_first", "bottom_pred"]:
        day_returns = []
        trades = 0
        for date, day in predictions.groupby("trade_date", sort=True):
            if mode == "top_pred":
                selected = day.sort_values(["pred_ret", "pred_win_prob", "code"], ascending=[False, False, True]).head(max_positions)
            elif mode == "bottom_pred":
                selected = day.sort_values(["pred_ret", "pred_win_prob", "code"], ascending=[True, True, True]).head(max_positions)
            elif mode == "code_first":
                selected = day.sort_values("code").head(max_positions)
            elif mode == "random_fixed":
                selected = day.sample(n=min(max_positions, len(day)), random_state=int(str(date).replace("-", "")))
            else:
                selected = day
            trades += int(len(selected))
            weight = total_exposure / len(selected) if len(selected) else 0.0
            day_returns.append(float((pd.to_numeric(selected["target_ret_net"], errors="coerce").fillna(0.0) * weight).sum()))
        nav = pd.DataFrame({"trade_date": sorted(predictions["trade_date"].astype(str).unique()), "nav": _nav_from_returns(day_returns)})
        rows.append({"mode": mode, **_portfolio_metrics(nav), "trades": trades, "avg_day_return": float(np.mean(day_returns)) if day_returns else 0.0})
    return pd.DataFrame(rows).sort_values("annual_return", ascending=False).reset_index(drop=True)


def prediction_deciles(predictions: pd.DataFrame) -> pd.DataFrame:
    out = predictions.copy()
    out["pred_ret_decile"] = pd.qcut(out["pred_ret"].rank(method="first"), 10, labels=False) + 1
    return (
        out.groupby("pred_ret_decile", as_index=False)
        .agg(
            rows=("target_ret_net", "size"),
            pred_ret_mean=("pred_ret", "mean"),
            realized_ret_mean=("target_ret_net", "mean"),
            win_rate=("target_win", "mean"),
            second_board_rate=("second_board_success", "mean"),
        )
        .sort_values("pred_ret_decile")
    )


def _nav_from_returns(returns: list[float]) -> list[float]:
    nav = [1.0]
    for ret in returns:
        nav.append(nav[-1] * (1.0 + float(ret)))
    return nav[1:]


def _variants() -> list[OvernightVariant]:
    return [
        OvernightVariant("top5_w01_total05", max_positions=5, max_name_weight=0.01, max_total_exposure=0.05),
        OvernightVariant("top5_w02_total10", max_positions=5, max_name_weight=0.02, max_total_exposure=0.10),
        OvernightVariant("top5_w04_total20", max_positions=5, max_name_weight=0.04, max_total_exposure=0.20),
        OvernightVariant("top3_w03_total09", max_positions=3, max_name_weight=0.03, max_total_exposure=0.09),
        OvernightVariant("top5_w10_full", max_positions=5, max_name_weight=0.10, max_total_exposure=0.50),
        OvernightVariant("top3_w15_full", max_positions=3, max_name_weight=0.15, max_total_exposure=0.45),
        OvernightVariant("top2_w20_full", max_positions=2, max_name_weight=0.20, max_total_exposure=0.40),
        OvernightVariant("top5_w10_pred_gt1pct", max_positions=5, max_name_weight=0.10, max_total_exposure=0.50, min_pred_return=0.01),
        OvernightVariant("top5_w10_prob_gt60", max_positions=5, max_name_weight=0.10, max_total_exposure=0.50, min_pred_win_prob=0.60),
        OvernightVariant("top5_w10_heat20_200", max_positions=5, max_name_weight=0.10, max_total_exposure=0.50, market_rule="heat", zero_below=20, half_below=200),
        OvernightVariant("top5_w10_prevup30_40", max_positions=5, max_name_weight=0.10, max_total_exposure=0.50, market_rule="prev_up", zero_below=0.30, half_below=0.40),
        OvernightVariant("top5_w10_heat_half_weak", max_positions=5, max_name_weight=0.10, max_total_exposure=0.50, market_rule="heat_half_weak", zero_below=20, half_below=200),
    ]


def _feature_columns(data: pd.DataFrame) -> list[str]:
    cols = [
        "ret_1",
        "intraday_ret",
        "turnover_rate",
        "amount_log",
        "adv20_log",
        "is_limit_10pct",
        "is_limit_20pct",
        "is_limit_unknown",
        "prev_up_ratio",
        "prev_avg_ret",
        "prev_sealed_count",
        "prev_failed_count",
        "prev_touch_count",
        "prev_seal_rate_among_touched",
    ]
    return [col for col in cols if col in data.columns]


def _fit_regressor(x: pd.DataFrame, y: pd.Series, random_state: int):
    x = x.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y = pd.to_numeric(y, errors="coerce").fillna(0.0)
    try:
        from lightgbm import LGBMRegressor

        model = LGBMRegressor(n_estimators=160, learning_rate=0.05, num_leaves=31, subsample=0.8, colsample_bytree=0.8, random_state=random_state, verbose=-1, n_jobs=-1)
        model.fit(x, y)
        return model
    except Exception:
        from sklearn.ensemble import HistGradientBoostingRegressor

        model = HistGradientBoostingRegressor(max_iter=160, learning_rate=0.05, random_state=random_state)
        model.fit(x, y)
        return model


def _fit_classifier(x: pd.DataFrame, y: pd.Series, random_state: int):
    x = x.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y = pd.to_numeric(y, errors="coerce").fillna(0).astype(int)
    if y.nunique() < 2:
        return _ConstantClassifier(float(y.mean() if len(y) else 0.0))
    try:
        from lightgbm import LGBMClassifier

        model = LGBMClassifier(n_estimators=160, learning_rate=0.05, num_leaves=31, subsample=0.8, colsample_bytree=0.8, random_state=random_state, verbose=-1, n_jobs=-1)
        model.fit(x, y)
        return model
    except Exception:
        from sklearn.ensemble import HistGradientBoostingClassifier

        model = HistGradientBoostingClassifier(max_iter=160, learning_rate=0.05, random_state=random_state)
        model.fit(x, y)
        return model


def _predict_regression(model, x: pd.DataFrame) -> np.ndarray:
    return np.asarray(model.predict(x.apply(pd.to_numeric, errors="coerce").fillna(0.0)), dtype=float)


def _predict_probability(model, x: pd.DataFrame) -> np.ndarray:
    x_num = x.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(x_num)[:, 1], dtype=float)
    return np.asarray(model.predict(x_num), dtype=float)


class _ConstantClassifier:
    def __init__(self, probability: float) -> None:
        self.probability = float(probability)

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        p = np.full(len(x), self.probability)
        return np.vstack([1.0 - p, p]).T


def _market_exposure(variant: OvernightVariant, candidates: pd.DataFrame) -> float:
    if variant.market_rule == "none" or candidates.empty:
        return 1.0
    prev_up = float(pd.to_numeric(candidates["prev_up_ratio"], errors="coerce").dropna().iloc[0]) if candidates["prev_up_ratio"].notna().any() else 1.0
    prev_sealed = float(pd.to_numeric(candidates["prev_sealed_count"], errors="coerce").dropna().iloc[0]) if candidates["prev_sealed_count"].notna().any() else 0.0
    if variant.market_rule == "prev_up":
        if variant.zero_below is not None and prev_up < float(variant.zero_below):
            return 0.0
        if variant.half_below is not None and prev_up < float(variant.half_below):
            return 0.5
    if variant.market_rule == "heat":
        return 1.0 if float(variant.zero_below or 0) <= prev_sealed <= float(variant.half_below or 1e9) else 0.0
    if variant.market_rule == "heat_half_weak":
        if not (float(variant.zero_below or 0) <= prev_sealed <= float(variant.half_below or 1e9)):
            return 0.0
        return 0.5 if prev_up < 0.4 else 1.0
    return 1.0


def _portfolio_metrics(nav: pd.DataFrame) -> dict[str, float]:
    if nav.empty:
        return {"total_return": 0.0, "annual_return": 0.0, "max_drawdown": 0.0, "sharpe": 0.0, "calmar": 0.0}
    values = pd.to_numeric(nav["nav"], errors="coerce")
    total_return = float(values.iloc[-1] / values.iloc[0] - 1.0) if len(values) > 1 and values.iloc[0] else 0.0
    daily = values.pct_change().dropna()
    max_drawdown = float((values / values.cummax() - 1.0).min())
    years = max(len(values) / 252.0, 1e-9)
    annual_return = float((1.0 + total_return) ** (1.0 / years) - 1.0) if total_return > -1 else -1.0
    sharpe = float(daily.mean() / daily.std(ddof=0) * np.sqrt(252)) if len(daily) and daily.std(ddof=0) else 0.0
    calmar = float(annual_return / abs(max_drawdown)) if max_drawdown < 0 else 0.0
    return {"total_return": total_return, "annual_return": annual_return, "max_drawdown": max_drawdown, "sharpe": sharpe, "calmar": calmar}


def _yearly_metrics(nav: pd.DataFrame) -> list[dict[str, float | int]]:
    if nav.empty:
        return []
    out = nav.copy()
    out["year"] = out["trade_date"].astype(str).str[:4].astype(int)
    return [{"year": int(year), **_portfolio_metrics(group)} for year, group in out.groupby("year", sort=True)]


if __name__ == "__main__":
    main()
