from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import pandas as pd

from ml_stock_selector.serving.live_sim import (
    PROFIT_PROTECT_RUN_ID,
    live_sim_reproducibility_snapshot,
    profit_protect_live_sim_config,
)


DEFAULT_STRATEGY_REPORT_DIR = Path("outputs/ml/reports/profit_protect_continuous_wf_2020_2025_20260622")
DEFAULT_MODEL_MANIFEST_ROOT = Path("outputs/ml/cache/folds_ret5_fundamental_fixed_rounds_20260621")


@dataclass(frozen=True)
class StrategyReport:
    summary: pd.DataFrame
    yearly: pd.DataFrame
    nav: pd.DataFrame
    orders: pd.DataFrame
    positions: pd.DataFrame
    diagnostics: pd.DataFrame
    exit_attribution: pd.DataFrame


def current_strategy_config() -> dict[str, object]:
    return live_sim_reproducibility_snapshot(profit_protect_live_sim_config())


def strategy_report_paths(report_dir: Path | str = DEFAULT_STRATEGY_REPORT_DIR) -> dict[str, Path]:
    root = Path(report_dir)
    return {
        "report_dir": root,
        "summary": root / "continuous_summary.csv",
        "yearly": root / "continuous_yearly.csv",
        "nav": root / "continuous_nav.csv",
        "orders": root / "continuous_orders.csv",
        "positions": root / "continuous_positions.csv",
        "diagnostics": root / "continuous_diagnostics.csv",
        "exit_attribution": root / "exit_reason_summary.csv",
    }


def current_strategy_report(report_dir: Path | str = DEFAULT_STRATEGY_REPORT_DIR) -> StrategyReport:
    paths = strategy_report_paths(report_dir)
    nav = _read_csv(paths["nav"])
    nav = _enrich_nav(nav)
    orders = _read_csv(paths["orders"])
    return StrategyReport(
        summary=_enrich_summary(_read_csv(paths["summary"]), nav),
        yearly=_enrich_yearly(_read_csv(paths["yearly"])),
        nav=nav,
        orders=orders,
        positions=_read_csv(paths["positions"]),
        diagnostics=_read_csv(paths["diagnostics"]),
        exit_attribution=_enrich_exit_attribution(_read_csv(paths["exit_attribution"])),
    )


def current_strategy_model_summary(
    manifest_root: Path | str = DEFAULT_MODEL_MANIFEST_ROOT,
    run_id: str = PROFIT_PROTECT_RUN_ID,
) -> pd.DataFrame:
    root = Path(manifest_root)
    rows: list[dict[str, object]] = []
    for manifest_path in sorted(root.glob(f"run_id={run_id}/fold_id=*/manifest.json")):
        manifest = _read_json(manifest_path)
        for role, artifact in dict(manifest.get("artifacts") or {}).items():
            params = _load_params_json(artifact)
            rows.append(
                {
                    "run_id": manifest.get("run_id"),
                    "fold_id": manifest.get("fold_id"),
                    "model_role": role,
                    "model_id": artifact.get("model_id"),
                    "model_type": artifact.get("model_type"),
                    "feature_set_id": artifact.get("feature_set_id") or manifest.get("feature_set_id"),
                    "label_name": artifact.get("label_name"),
                    "label_base": artifact.get("label_base") or manifest.get("label_base"),
                    "horizon_d": artifact.get("horizon_d") or manifest.get("horizon_d"),
                    "train_start": manifest.get("train_start"),
                    "train_end": manifest.get("train_end"),
                    "valid_start": manifest.get("valid_start"),
                    "valid_end": manifest.get("valid_end"),
                    "test_start": manifest.get("test_start"),
                    "test_end": manifest.get("test_end"),
                    "train_rows": (artifact.get("metrics") or {}).get("train_rows") or manifest.get("train_rows"),
                    "fixed_alpha_rounds": manifest.get("fixed_alpha_rounds"),
                    "fixed_risk_rounds": manifest.get("fixed_risk_rounds"),
                    "train_window_mode": manifest.get("train_window_mode"),
                    "source_train_window_mode": manifest.get("source_train_window_mode"),
                    "model_mode": manifest.get("model_mode"),
                    "alpha_eval_metric": manifest.get("alpha_eval_metric"),
                    "alpha_eval_target": manifest.get("alpha_eval_target"),
                    "n_estimators": params.get("n_estimators"),
                    "objective": params.get("objective"),
                    "metric": params.get("metric"),
                    "learning_rate": params.get("learning_rate"),
                    "num_leaves": params.get("num_leaves"),
                    "min_data_in_leaf": params.get("min_data_in_leaf"),
                    "early_stopping_rounds": params.get("early_stopping_rounds"),
                    "artifact_uri": artifact.get("artifact_uri"),
                    "feature_schema_uri": artifact.get("feature_schema_uri"),
                    "manifest_path": str(manifest_path),
                }
            )
    return pd.DataFrame(rows)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _read_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_params_json(artifact: dict[str, object]) -> dict[str, object]:
    artifact_uri = artifact.get("artifact_uri")
    if not artifact_uri:
        return {}
    params_path = Path(str(artifact_uri)).with_suffix(".params.json")
    if not params_path.exists():
        return {}
    return _read_json(params_path)


def _enrich_nav(nav: pd.DataFrame) -> pd.DataFrame:
    if nav.empty or "nav" not in nav:
        return nav.assign(daily_return=pd.Series(dtype=float), drawdown=pd.Series(dtype=float))
    out = nav.copy()
    out["sim_date"] = pd.to_datetime(out["sim_date"], errors="coerce")
    out = out.sort_values("sim_date").reset_index(drop=True)
    out["nav"] = pd.to_numeric(out["nav"], errors="coerce")
    out["daily_return"] = out["nav"].pct_change().fillna(0.0)
    out["drawdown"] = out["nav"] / out["nav"].cummax() - 1.0
    if "account_drawdown" not in out:
        out["account_drawdown"] = out["drawdown"]
    return out


def _enrich_summary(summary: pd.DataFrame, nav: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary
    out = summary.copy()
    returns = pd.to_numeric(nav.get("daily_return", pd.Series(dtype=float)), errors="coerce").dropna()
    if "volatility" not in out:
        out["volatility"] = _annualized_volatility(returns)
    if "sharpe" not in out:
        out["sharpe"] = _sharpe(returns)
    if "sortino" not in out:
        out["sortino"] = _sortino(returns)
    if "calmar" not in out:
        annual_return = float(pd.to_numeric(out.get("annual_return", pd.Series([0.0])), errors="coerce").iloc[0] or 0.0)
        max_drawdown = abs(float(pd.to_numeric(out.get("max_drawdown", pd.Series([0.0])), errors="coerce").iloc[0] or 0.0))
        out["calmar"] = annual_return / max_drawdown if max_drawdown > 0 else 0.0
    return out


def _enrich_yearly(yearly: pd.DataFrame) -> pd.DataFrame:
    if yearly.empty:
        return yearly
    out = yearly.copy()
    if "calmar" not in out and {"total_return", "max_drawdown"}.issubset(out.columns):
        drawdown = pd.to_numeric(out["max_drawdown"], errors="coerce").abs()
        returns = pd.to_numeric(out["total_return"], errors="coerce")
        out["calmar"] = returns.where(drawdown > 0, 0.0) / drawdown.where(drawdown > 0, 1.0)
    return out


def _enrich_exit_attribution(attribution: pd.DataFrame) -> pd.DataFrame:
    if attribution.empty or "total_realized_pnl" not in attribution:
        return attribution
    out = attribution.copy()
    total_loss = pd.to_numeric(out["total_realized_pnl"], errors="coerce").clip(upper=0).abs().sum()
    if total_loss > 0:
        out["loss_contribution"] = pd.to_numeric(out["total_realized_pnl"], errors="coerce").clip(upper=0).abs() / total_loss
    else:
        out["loss_contribution"] = 0.0
    return out.sort_values("total_realized_pnl", ascending=True).reset_index(drop=True)


def _annualized_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    if returns.empty:
        return 0.0
    return float(returns.std(ddof=0) * periods_per_year**0.5)


def _sharpe(returns: pd.Series, periods_per_year: int = 252) -> float:
    volatility = returns.std(ddof=0)
    if returns.empty or volatility == 0:
        return 0.0
    return float(returns.mean() / volatility * periods_per_year**0.5)


def _sortino(returns: pd.Series, periods_per_year: int = 252) -> float:
    downside = returns[returns < 0]
    downside_volatility = downside.std(ddof=0)
    if returns.empty or downside_volatility == 0:
        return 0.0
    return float(returns.mean() / downside_volatility * periods_per_year**0.5)
