from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd

from ml_stock_selector.portfolio.constraints import FixedHorizonRiskFilterConfig
from ml_stock_selector.portfolio.holding_policy import HoldingState, holding_state_from_row
from ml_stock_selector.universe import detect_is_bse


TARGET_COLUMNS = [
    "trade_date",
    "portfolio_id",
    "code",
    "target_weight",
    "rank_n",
    "trade_score",
    "absolute_rank_pct",
    "risk_rank_pct",
    "entry_abs_rank_pct",
    "entry_risk_rank_pct",
    "entry_reason",
    "signal_action",
    "hold_reason",
    "exit_reason",
    "sell_blocked_reason",
    "entry_date",
    "entry_price",
    "shares",
    "holding_days",
    "entry_trade_score",
    "latest_trade_score",
    "generated_at",
]

DIAGNOSTIC_COLUMNS = [
    "trade_date",
    "strategy_id",
    "raw_candidate_count",
    "hard_filter_pass_count",
    "risk_entry_rejected_count",
    "abs_rank_rejected_count",
    "candidate_pool_size",
    "final_selected_count",
    "retained_holdings_count",
    "buy_count",
    "sell_count",
    "risk_exit_count",
    "time_exit_count",
    "sell_blocked_count",
    "avg_entry_abs_rank_pct",
    "avg_entry_risk_rank_pct",
    "cash_weight",
    "created_at",
]


@dataclass(frozen=True)
class PortfolioTargetResult:
    targets: pd.DataFrame
    diagnostics: pd.DataFrame


def construct_fixed_5d_risk_filter_targets(
    scored_candidates: pd.DataFrame,
    current_holdings: list[HoldingState] | pd.DataFrame,
    constraints: FixedHorizonRiskFilterConfig,
    trade_date: str,
) -> PortfolioTargetResult:
    raw = scored_candidates.copy()
    holdings = _normalize_holdings(current_holdings)
    latest_by_code = _latest_rows_by_code(raw)
    rows: list[dict[str, object]] = []
    held_active: set[str] = set()
    stats = {"sell_count": 0, "risk_exit_count": 0, "time_exit_count": 0, "sell_blocked_count": 0}
    entry_pool, hard_pass_count, risk_rejected, abs_rejected = _entry_pool(raw, constraints)
    renewal_rank_by_code = _candidate_rank_by_code(entry_pool)

    for holding in holdings:
        latest = latest_by_code.get(holding.code, pd.Series({"trade_date": trade_date, "code": holding.code}))
        row, active = _row_for_holding(
            holding,
            latest,
            constraints,
            trade_date,
            stats,
            renewal_rank_by_code.get(holding.code),
        )
        rows.append(row)
        if active:
            held_active.add(holding.code)

    selected = _select_entries(entry_pool, held_active, holdings, constraints)
    rows.extend(_entry_rows(selected, constraints, trade_date))

    targets = _targets_frame(rows, constraints, trade_date)
    diagnostics = _diagnostics_frame(
        raw,
        targets,
        constraints,
        trade_date,
        hard_pass_count,
        risk_rejected,
        abs_rejected,
        stats,
    )
    return PortfolioTargetResult(targets=targets, diagnostics=diagnostics)


def fixed_horizon_config_from_dict(raw: dict[str, object] | None) -> FixedHorizonRiskFilterConfig:
    if not raw:
        return FixedHorizonRiskFilterConfig()
    fields = FixedHorizonRiskFilterConfig.__dataclass_fields__
    values = {key: raw[key] for key in fields if key in raw}
    return FixedHorizonRiskFilterConfig(**values)


def _normalize_holdings(current_holdings: list[HoldingState] | pd.DataFrame) -> list[HoldingState]:
    if isinstance(current_holdings, pd.DataFrame):
        if current_holdings.empty:
            return []
        return [holding_state_from_row(pd.Series(row._asdict())) for row in current_holdings.itertuples(index=False)]
    return list(current_holdings or [])


def _row_for_holding(
    holding: HoldingState,
    latest: pd.Series,
    constraints: FixedHorizonRiskFilterConfig,
    trade_date: str,
    stats: dict[str, int],
    renewal_rank: int | None = None,
) -> tuple[dict[str, object], bool]:
    renewed = _is_renewed_holding(holding, constraints, renewal_rank)
    exit_reason = _exit_reason(holding, latest, constraints, renewed)
    can_sell = _optional_bool(latest.get("can_sell_next_open"), True)
    if exit_reason is None:
        action = "hold"
        active = True
        hold_reason = "renewed_top_candidate" if renewed else "fixed_horizon_not_due"
        sell_blocked_reason = None
        target_weight = 1.0
    elif can_sell:
        action = "sell"
        active = False
        hold_reason = None
        sell_blocked_reason = None
        target_weight = 0.0
        stats["sell_count"] += 1
        if exit_reason == "risk_exit":
            stats["risk_exit_count"] += 1
        elif exit_reason == "time_exit":
            stats["time_exit_count"] += 1
    else:
        action = "sell_blocked"
        active = True
        hold_reason = None
        sell_blocked_reason = exit_reason
        target_weight = 1.0
        stats["sell_blocked_count"] += 1
    return (
        {
            "trade_date": trade_date,
            "portfolio_id": constraints.strategy_id,
            "code": holding.code,
            "target_weight": target_weight,
            "trade_score": _optional_float(latest.get("absolute_rank_pct", holding.latest_trade_score)) or 0.0,
            "absolute_rank_pct": _optional_float(latest.get("absolute_rank_pct")),
            "risk_rank_pct": _optional_float(latest.get("risk_rank_pct")),
            "entry_abs_rank_pct": _optional_float(getattr(holding, "entry_trade_score", None)),
            "entry_risk_rank_pct": _optional_float(_series_get(latest, "entry_risk_rank_pct")),
            "entry_reason": holding.entry_reason,
            "signal_action": action,
            "hold_reason": hold_reason,
            "exit_reason": exit_reason,
            "sell_blocked_reason": sell_blocked_reason,
            "entry_date": holding.entry_date,
            "entry_price": holding.entry_price,
            "shares": holding.shares,
            "holding_days": holding.holding_days,
            "entry_trade_score": holding.entry_trade_score,
            "latest_trade_score": _optional_float(latest.get("absolute_rank_pct", holding.latest_trade_score)),
            "generated_at": _now(),
        },
        active,
    )


def _exit_reason(
    holding: HoldingState,
    latest: pd.Series,
    constraints: FixedHorizonRiskFilterConfig,
    renewed: bool = False,
) -> str | None:
    if constraints.enable_risk_exit:
        risk_rank = _optional_float(latest.get("risk_rank_pct"))
        if risk_rank is not None and risk_rank >= constraints.risk_exit_rank_pct:
            return "risk_exit"
    if renewed:
        return None
    if constraints.enable_time_exit and holding.holding_days >= constraints.holding_days:
        return "time_exit"
    return None


def _is_renewed_holding(
    holding: HoldingState,
    constraints: FixedHorizonRiskFilterConfig,
    renewal_rank: int | None,
) -> bool:
    return (
        constraints.enable_time_exit
        and holding.holding_days >= constraints.holding_days
        and renewal_rank is not None
        and renewal_rank <= constraints.renewal_candidate_rank
    )


def _entry_pool(
    raw: pd.DataFrame,
    constraints: FixedHorizonRiskFilterConfig,
) -> tuple[pd.DataFrame, int, int, int]:
    if raw.empty:
        return raw.copy(), 0, 0, 0
    out = raw.copy()
    hard = pd.Series(True, index=out.index)
    if constraints.exclude_bse:
        if "is_bse" in out:
            hard &= ~out["is_bse"].fillna(False).astype(bool)
        elif "code" in out:
            hard &= ~out["code"].map(detect_is_bse)
    if constraints.exclude_st and "is_st" in out:
        hard &= ~out["is_st"].fillna(False).astype(bool)
    if constraints.exclude_paused and "is_paused" in out:
        hard &= ~out["is_paused"].fillna(False).astype(bool)
    if constraints.require_can_buy_next_open and "can_buy_next_open" in out:
        hard &= out["can_buy_next_open"].fillna(False).astype(bool)
    if "adv20_amount" in out:
        hard &= pd.to_numeric(out["adv20_amount"], errors="coerce").fillna(0.0) >= constraints.min_adv20_amount
    hard_filtered = out[hard].copy()
    risk_mask = pd.to_numeric(hard_filtered.get("risk_rank_pct", pd.Series(1.0, index=hard_filtered.index)), errors="coerce").fillna(1.0) <= constraints.risk_entry_max_rank_pct
    risk_filtered = hard_filtered[risk_mask].copy()
    abs_mask = pd.to_numeric(risk_filtered.get("absolute_rank_pct", pd.Series(0.0, index=risk_filtered.index)), errors="coerce").fillna(0.0) >= constraints.min_abs_rank_pct
    final = risk_filtered[abs_mask].copy()
    return final, len(hard_filtered), int((~risk_mask).sum()), int((~abs_mask).sum())


def _candidate_rank_by_code(entry_pool: pd.DataFrame) -> dict[str, int]:
    if entry_pool.empty or "code" not in entry_pool:
        return {}
    ordered = entry_pool.sort_values(["absolute_rank_pct", "code"], ascending=[False, True]).reset_index(drop=True)
    return {str(row.code): int(idx + 1) for idx, row in enumerate(ordered.itertuples(index=False))}


def _select_entries(
    entry_pool: pd.DataFrame,
    held_active: set[str],
    holdings: list[HoldingState],
    constraints: FixedHorizonRiskFilterConfig,
) -> pd.DataFrame:
    if entry_pool.empty:
        return entry_pool.copy()
    held_codes = {holding.code for holding in holdings}
    capacity = min(constraints.target_positions, constraints.hard_max_positions) - len(held_active)
    if capacity <= 0:
        return entry_pool.iloc[0:0].copy()
    new_limit = constraints.max_initial_entries if not held_codes else constraints.max_new_entries_per_day
    limit = max(0, min(capacity, new_limit))
    # Do not reopen a name on the same decision date that it is being exited.
    # The fixed-horizon baseline should produce at most one target row per code
    pool = entry_pool[~entry_pool["code"].astype(str).isin(held_codes)].copy()
    return pool.sort_values(["absolute_rank_pct", "code"], ascending=[False, True]).head(limit)


def _entry_rows(selected: pd.DataFrame, constraints: FixedHorizonRiskFilterConfig, trade_date: str) -> list[dict[str, object]]:
    rows = []
    for rank, (_, row) in enumerate(selected.iterrows(), start=1):
        abs_rank = _optional_float(row.get("absolute_rank_pct")) or 0.0
        rows.append(
            {
                "trade_date": trade_date,
                "portfolio_id": constraints.strategy_id,
                "code": str(row["code"]),
                "target_weight": 1.0,
                "rank_n": rank,
                "trade_score": abs_rank,
                "absolute_rank_pct": abs_rank,
                "risk_rank_pct": _optional_float(row.get("risk_rank_pct")),
                "entry_abs_rank_pct": abs_rank,
                "entry_risk_rank_pct": _optional_float(row.get("risk_rank_pct")),
                "entry_reason": "fixed_5d_abs_rank",
                "signal_action": "buy",
                "hold_reason": None,
                "exit_reason": None,
                "sell_blocked_reason": None,
                "entry_date": None,
                "entry_price": None,
                "shares": None,
                "holding_days": 0,
                "entry_trade_score": abs_rank,
                "latest_trade_score": abs_rank,
                "generated_at": _now(),
            }
        )
    return rows


def _targets_frame(rows: list[dict[str, object]], constraints: FixedHorizonRiskFilterConfig, trade_date: str) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=TARGET_COLUMNS)
    out = pd.DataFrame(rows)
    out["rank_n"] = range(1, len(out) + 1)
    active_mask = out["signal_action"].astype(str) != "sell"
    active_count = int(active_mask.sum())
    out.loc[~active_mask, "target_weight"] = 0.0
    if active_count:
        weight = min(max(1.0 / active_count, constraints.min_position_weight), constraints.max_position_weight)
        out.loc[active_mask, "target_weight"] = weight
    out["trade_date"] = trade_date
    return out[[column for column in TARGET_COLUMNS if column in out.columns]].copy()


def _diagnostics_frame(
    raw: pd.DataFrame,
    targets: pd.DataFrame,
    constraints: FixedHorizonRiskFilterConfig,
    trade_date: str,
    hard_pass_count: int,
    risk_rejected: int,
    abs_rejected: int,
    stats: dict[str, int],
) -> pd.DataFrame:
    active = targets[targets["signal_action"].astype(str) != "sell"] if not targets.empty else targets
    entries = targets[targets["signal_action"].astype(str) == "buy"] if not targets.empty else targets
    target_positions = max(constraints.target_positions, 1)
    cash_weight = max(0.0, 1.0 - min(len(active), target_positions) / target_positions)
    row = {
        "trade_date": trade_date,
        "strategy_id": constraints.strategy_id,
        "raw_candidate_count": len(raw),
        "hard_filter_pass_count": hard_pass_count,
        "risk_entry_rejected_count": risk_rejected,
        "abs_rank_rejected_count": abs_rejected,
        "candidate_pool_size": int((pd.to_numeric(raw.get("absolute_rank_pct", pd.Series(dtype=float)), errors="coerce") >= constraints.min_abs_rank_pct).sum()) if not raw.empty else 0,
        "final_selected_count": len(active),
        "retained_holdings_count": int((targets.get("signal_action", pd.Series(dtype=object)) == "hold").sum()) if not targets.empty else 0,
        "buy_count": int((targets.get("signal_action", pd.Series(dtype=object)) == "buy").sum()) if not targets.empty else 0,
        "sell_count": stats["sell_count"],
        "risk_exit_count": stats["risk_exit_count"],
        "time_exit_count": stats["time_exit_count"],
        "sell_blocked_count": stats["sell_blocked_count"],
        "avg_entry_abs_rank_pct": _mean(entries, "entry_abs_rank_pct"),
        "avg_entry_risk_rank_pct": _mean(entries, "entry_risk_rank_pct"),
        "cash_weight": cash_weight,
        "created_at": _now(),
    }
    return pd.DataFrame([row], columns=DIAGNOSTIC_COLUMNS)


def _latest_rows_by_code(frame: pd.DataFrame) -> dict[str, pd.Series]:
    if frame.empty or "code" not in frame:
        return {}
    return {str(row["code"]): row for _, row in frame.drop_duplicates("code", keep="last").iterrows()}


def _mean(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame:
        return 0.0
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.mean()) if not values.empty else 0.0


def _optional_bool(value: object, default: bool) -> bool:
    if value is None or pd.isna(value):
        return default
    return bool(value)


def _optional_float(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _series_get(row: pd.Series, column: str) -> object | None:
    return row[column] if column in row else None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
