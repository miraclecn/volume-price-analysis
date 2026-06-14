from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path
import sys

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml_stock_selector.backtest.data_access import load_backtest_candidates
from ml_stock_selector.backtest.engine import BacktestConfig, run_holding_aware_backtest
from ml_stock_selector.backtest.execution import ExecutionConfig
from ml_stock_selector.backtest.metrics import max_drawdown, summarize_fold_metric_rows
from ml_stock_selector.backtest.reports import portfolio_diagnostics_report_metrics
from ml_stock_selector.backtest.walkforward import _portfolio_constraints_from_config
from ml_stock_selector.config import load_ml_config
from ml_stock_selector.data_access import load_normalized_stock_bars


PREFERRED_RUN_ID = "wf_three_model_v2_adv10m_001"
PREFERRED_PREDICTION_SCORE_VERSION = "v2_three_model"
PREFERRED_PORTFOLIO_SCORE_VERSION = "v2_abs_risk_lowadv_full015_top12"
PREFERRED_STRATEGY_PREFIX = "absolute_risk_filter_score_adv_combo_top12"


@dataclass(frozen=True)
class RiskVariant:
    label: str
    candidate_risk_max_rank_pct: float
    core_risk_max_rank_pct: float


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/ml_walkforward_adv10m.toml")
    parser.add_argument("--run-id", default=PREFERRED_RUN_ID)
    parser.add_argument("--start-year", type=int, default=2020)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument(
        "--variant",
        action="append",
        default=[],
        help="Risk variant as label:candidate_threshold:core_threshold, e.g. risk050:0.50:0.50.",
    )
    parser.add_argument("--adv-weight", type=float, default=0.15)
    parser.add_argument("--output-dir", default="outputs/ml/risk_sweep")
    return parser


def parse_variant(text: str) -> RiskVariant:
    parts = text.split(":")
    if len(parts) != 3:
        raise ValueError("variant must be label:candidate_threshold:core_threshold")
    label, candidate, core = parts
    return RiskVariant(label, float(candidate), float(core))


def default_variants(values: list[str]) -> list[RiskVariant]:
    if values:
        return [parse_variant(value) for value in values]
    return [RiskVariant("risk050", 0.50, 0.50)]


def build_portfolio_id(fold_id: str, variant: RiskVariant) -> str:
    return (
        f"{fold_id}_{PREFERRED_STRATEGY_PREFIX}_{variant.label}"
        f"_c{_threshold_token(variant.candidate_risk_max_rank_pct)}"
        f"_core{_threshold_token(variant.core_risk_max_rank_pct)}"
    )


def apply_preferred_adv015_score(candidates: pd.DataFrame, *, adv_weight: float = 0.15) -> pd.DataFrame:
    scored = candidates.copy()
    scored["absolute_rank_pct"] = pd.to_numeric(scored["absolute_rank_pct"], errors="coerce").fillna(0.0)
    scored["risk_rank_pct"] = pd.to_numeric(scored["risk_rank_pct"], errors="coerce").fillna(1.0)
    scored["alpha_rank_pct"] = scored["absolute_rank_pct"]
    scored["active_rank_pct"] = scored["absolute_rank_pct"]
    scored["full_prediction_pool_adv_pct"] = (
        scored.groupby("trade_date")["adv20_amount"]
        .rank(method="average", pct=True, ascending=True)
        .fillna(1.0)
    )
    scored["trade_score_v2"] = (
        (1.0 - adv_weight) * scored["absolute_rank_pct"]
        + adv_weight * (1.0 - scored["full_prediction_pool_adv_pct"])
    )
    scored["core_score"] = scored["trade_score_v2"]
    scored["trade_score"] = scored["trade_score_v2"]
    scored["score_version"] = PREFERRED_PORTFOLIO_SCORE_VERSION
    return scored


def build_constraints(config, variant: RiskVariant):
    return replace(
        _portfolio_constraints_from_config(config),
        target_positions=12,
        hard_max_positions=15,
        max_initial_entries=12,
        candidate_risk_max_rank_pct=variant.candidate_risk_max_rank_pct,
        core_risk_max_rank_pct=variant.core_risk_max_rank_pct,
        min_adv20_amount=10_000_000,
    )


def continuous_nav_summary(nav: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    if nav.empty:
        return nav.copy(), {"total_return": 0.0, "annual_return": 0.0, "max_drawdown": 0.0}
    ordered = nav.sort_values(["sim_date", "fold_id"]).reset_index(drop=True).copy()
    stitched_parts: list[pd.DataFrame] = []
    current_nav: float | None = None
    for _, fold_nav in ordered.groupby("fold_id", sort=False):
        fold_nav = fold_nav.sort_values("sim_date").copy()
        first_nav = float(pd.to_numeric(fold_nav["nav"], errors="coerce").iloc[0])
        scale = 1.0 if current_nav is None else current_nav / first_nav
        fold_nav["continuous_nav"] = pd.to_numeric(fold_nav["nav"], errors="coerce").astype(float) * scale
        current_nav = float(fold_nav["continuous_nav"].iloc[-1])
        stitched_parts.append(fold_nav)
    stitched = pd.concat(stitched_parts, ignore_index=True)
    values = pd.to_numeric(stitched["continuous_nav"], errors="coerce").astype(float)
    total_return = float(values.iloc[-1] / values.iloc[0] - 1.0) if len(values) >= 2 else 0.0
    annual_return = _annualized_by_dates(stitched, "continuous_nav")
    summary = {
        "total_return": total_return,
        "annual_return": annual_return,
        "max_drawdown": max_drawdown(stitched, nav_col="continuous_nav"),
    }
    return stitched, summary


def main() -> None:
    args = build_arg_parser().parse_args()
    config = load_ml_config(args.config)
    variants = default_variants(args.variant)
    folds = [
        fold
        for fold in config.split["folds"]
        if args.start_year <= _fold_year(str(fold["fold_id"])) <= args.end_year
    ]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_metrics: list[dict[str, object]] = []
    all_nav: list[pd.DataFrame] = []
    all_continuous: list[dict[str, object]] = []

    ml_con = duckdb.connect(str(config.data["ml_db"]), read_only=True)
    try:
        for variant in variants:
            variant_nav: list[pd.DataFrame] = []
            constraints = build_constraints(config, variant)
            for fold in folds:
                fold_id = str(fold["fold_id"])
                portfolio_id = build_portfolio_id(fold_id, variant)
                print(
                    f"Running {fold_id} {variant.label} "
                    f"candidate<={variant.candidate_risk_max_rank_pct:.2f} "
                    f"core<={variant.core_risk_max_rank_pct:.2f}",
                    flush=True,
                )
                candidates = load_backtest_candidates(
                    ml_con,
                    run_id=args.run_id,
                    fold_id=fold_id,
                    score_version=PREFERRED_PREDICTION_SCORE_VERSION,
                    exclude_bse=True,
                )
                if candidates.empty:
                    raise ValueError(f"No predictions found for {args.run_id}/{fold_id}")
                scored = apply_preferred_adv015_score(candidates, adv_weight=float(args.adv_weight))
                bars = load_normalized_stock_bars(
                    str(config.data["alpha_data_db"]),
                    str(fold["test_start"]),
                    str(fold["test_end"]),
                    str(config.data["normalized_bars_table"]),
                )
                result = run_holding_aware_backtest(
                    scored,
                    bars,
                    constraints,
                    BacktestConfig(
                        float(config.backtest["initial_cash"]),
                        fold_id,
                        ExecutionConfig(),
                        decision_dates=sorted(scored["trade_date"].dropna().unique()),
                    ),
                    min_weight=float(config.portfolio["single_name_min_weight"]),
                    max_weight=float(config.portfolio["single_name_max_weight"]),
                    allow_cash=bool(config.portfolio["allow_cash"]),
                    run_id=f"{args.run_id}_{variant.label}",
                    fold_id=portfolio_id,
                    score_version=PREFERRED_PORTFOLIO_SCORE_VERSION,
                )
                diagnostics = portfolio_diagnostics_report_metrics(result.portfolio_diagnostics)
                metric_rows = summarize_fold_metric_rows(
                    result,
                    run_id=f"{args.run_id}_{variant.label}",
                    fold_id=portfolio_id,
                    score_version=PREFERRED_PORTFOLIO_SCORE_VERSION,
                    strategy_id=portfolio_id,
                    start_date=str(scored["trade_date"].min()),
                    end_date=str(scored["trade_date"].max()),
                    candidate_pool_size=diagnostics["avg_candidate_pool_size"],
                    core_pool_size=diagnostics["avg_core_pool_size"],
                )
                metric = pivot_metric_rows(metric_rows)
                metric.update(
                    {
                        "period": str(_fold_year(fold_id)),
                        "source_run_id": args.run_id,
                        "source_fold_id": fold_id,
                        "variant": variant.label,
                        "candidate_risk_max_rank_pct": variant.candidate_risk_max_rank_pct,
                        "core_risk_max_rank_pct": variant.core_risk_max_rank_pct,
                        "adv_weight": float(args.adv_weight),
                        "prediction_rows": len(scored),
                        "avg_candidate_pool": diagnostics["avg_candidate_pool_size"],
                        "avg_core_pool": diagnostics["avg_core_pool_size"],
                    }
                )
                all_metrics.append(metric)
                nav = result.nav.copy()
                nav["period"] = str(_fold_year(fold_id))
                nav["source_fold_id"] = fold_id
                nav["fold_id"] = portfolio_id
                nav["variant"] = variant.label
                variant_nav.append(nav)
                all_nav.append(nav)

            stitched, summary = continuous_nav_summary(pd.concat(variant_nav, ignore_index=True))
            stitched["variant"] = variant.label
            stitched.to_csv(output_dir / f"preferred_adv10m_{variant.label}_continuous_nav.csv", index=False)
            all_continuous.append(
                {
                    "variant": variant.label,
                    "source_run_id": args.run_id,
                    "start_year": args.start_year,
                    "end_year": args.end_year,
                    "candidate_risk_max_rank_pct": variant.candidate_risk_max_rank_pct,
                    "core_risk_max_rank_pct": variant.core_risk_max_rank_pct,
                    "adv_weight": float(args.adv_weight),
                    **summary,
                }
            )
    finally:
        ml_con.close()

    metrics = pd.DataFrame(all_metrics)
    nav = pd.concat(all_nav, ignore_index=True) if all_nav else pd.DataFrame()
    continuous = pd.DataFrame(all_continuous)
    metrics_path = output_dir / "preferred_adv10m_risk_sweep_metrics.csv"
    nav_path = output_dir / "preferred_adv10m_risk_sweep_nav.csv"
    continuous_path = output_dir / "preferred_adv10m_risk_sweep_continuous_summary.csv"
    metrics.to_csv(metrics_path, index=False)
    nav.to_csv(nav_path, index=False)
    continuous.to_csv(continuous_path, index=False)
    print(f"Wrote {metrics_path}", flush=True)
    print(f"Wrote {nav_path}", flush=True)
    print(f"Wrote {continuous_path}", flush=True)


def pivot_metric_rows(metric_rows: pd.DataFrame) -> dict[str, object]:
    context = metric_rows.iloc[0][["run_id", "fold_id", "start_date", "end_date"]].to_dict()
    values = metric_rows.set_index("metric_name")["metric_value"].to_dict()
    return {**context, **values}


def _threshold_token(value: float) -> str:
    return f"{int(round(value * 100)):03d}"


def _fold_year(fold_id: str) -> int:
    if not fold_id.startswith("wf_"):
        raise ValueError(f"Unsupported fold_id: {fold_id}")
    return int(fold_id.removeprefix("wf_")[:4])


def _annualized_by_dates(nav: pd.DataFrame, nav_col: str) -> float:
    if len(nav) < 2:
        return 0.0
    ordered = nav.sort_values("sim_date")
    values = pd.to_numeric(ordered[nav_col], errors="coerce").astype(float)
    start = float(values.iloc[0])
    end = float(values.iloc[-1])
    if start <= 0.0:
        return 0.0
    if end <= 0.0:
        return -1.0
    dates = pd.to_datetime(ordered["sim_date"], errors="coerce")
    years = (dates.iloc[-1] - dates.iloc[0]).days / 365.25
    if years <= 0.0:
        return 0.0
    return float((end / start) ** (1.0 / years) - 1.0)


if __name__ == "__main__":
    main()
