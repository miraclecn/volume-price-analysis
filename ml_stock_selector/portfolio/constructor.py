from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from ml_stock_selector.portfolio.constraints import (
    PortfolioConstraints,
    apply_hard_filters,
    is_unknown_industry,
)
from ml_stock_selector.portfolio.holding_policy import (
    SellDecision,
    evaluate_sell_decision,
    holding_state_from_row,
)
from ml_stock_selector.universe import detect_is_bse

PORTFOLIO_DIAGNOSTICS_ATTR = "portfolio_construction_diagnostics"

TARGET_COLUMNS = [
    "trade_date",
    "portfolio_id",
    "code",
    "industry_code",
    "industry_name",
    "target_weight",
    "rank_n",
    "trade_score",
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
    "run_id",
    "fold_id",
    "portfolio_id",
    "score_version",
    "raw_candidate_count",
    "hard_filter_pass_count",
    "core_pool_size",
    "candidate_pool_size",
    "selected_from_core",
    "selected_from_candidate",
    "final_selected_count",
    "low_adv_rejected_count",
    "cannot_buy_rejected_count",
    "st_rejected_count",
    "paused_rejected_count",
    "bse_rejected_count",
    "low_trade_score_rejected_count",
    "high_risk_rejected_count",
    "industry_limit_blocked_count",
    "unknown_industry_limit_blocked_count",
    "max_new_entries_blocked_count",
    "retained_holdings_count",
    "sell_signal_count",
    "sell_executed_count",
    "sell_blocked_count",
    "hold_due_to_min_days_count",
    "hold_due_to_score_ok_count",
    "exit_due_to_score_count",
    "exit_due_to_risk_count",
    "exit_due_to_time_count",
    "exit_due_to_not_candidate_count",
    "avg_holding_days_current",
    "median_holding_days_current",
    "cash_weight",
    "created_at",
]


def construct_portfolio_targets(
    scored_candidates: pd.DataFrame,
    constraints: PortfolioConstraints,
    portfolio_id: str,
    current_holdings: pd.DataFrame | None = None,
) -> pd.DataFrame:
    filtered = apply_hard_filters(scored_candidates, constraints)
    if "trade_date" in filtered and filtered["trade_date"].nunique(dropna=False) > 1:
        daily_targets = [
            _construct_portfolio_targets_for_frame(day, constraints, portfolio_id, current_holdings)
            for _, day in filtered.groupby("trade_date", sort=True)
        ]
        non_empty = [frame for frame in daily_targets if not frame.empty]
        if not non_empty:
            return pd.DataFrame(columns=TARGET_COLUMNS)
        return pd.concat(non_empty, ignore_index=True)
    return _construct_portfolio_targets_for_frame(filtered, constraints, portfolio_id, current_holdings)


def _construct_portfolio_targets_for_frame(
    filtered: pd.DataFrame,
    constraints: PortfolioConstraints,
    portfolio_id: str,
    current_holdings: pd.DataFrame | None = None,
) -> pd.DataFrame:
    selected_rows = []
    industry_counts: dict[object, int] = {}
    unknown_industry_count = 0
    new_entries = 0
    held = set(current_holdings["code"]) if current_holdings is not None and "code" in current_holdings else set()
    for row in filtered.sort_values(["trade_score", "code"], ascending=[False, True]).itertuples(index=False):
        code = getattr(row, "code")
        industry = getattr(row, "industry_code", None)
        if len(selected_rows) >= min(constraints.target_positions, constraints.hard_max_positions):
            break
        industry_unknown = is_unknown_industry(industry)
        if industry_unknown and unknown_industry_count >= constraints.max_unknown_industry_names:
            continue
        if not industry_unknown and industry_counts.get(industry, 0) >= constraints.max_industry_names:
            continue
        if code not in held and new_entries >= constraints.max_new_entries_per_day:
            continue
        if code not in held:
            new_entries += 1
        row_dict = row._asdict()
        if industry_unknown:
            unknown_industry_count += 1
        else:
            industry_counts[industry] = industry_counts.get(industry, 0) + 1
        selected_rows.append(row_dict)
    out = pd.DataFrame(selected_rows)
    if out.empty:
        return pd.DataFrame(columns=TARGET_COLUMNS)
    out["portfolio_id"] = portfolio_id
    out["rank_n"] = range(1, len(out) + 1)
    out["target_weight"] = 0.0
    out["entry_reason"] = [
        "trade_score; industry_unknown" if is_unknown_industry(value) else "trade_score"
        for value in out.get("industry_code", pd.Series([None] * len(out)))
    ]
    out["generated_at"] = datetime.now(timezone.utc).isoformat()
    return out[[column for column in TARGET_COLUMNS if column in out.columns]]


def get_portfolio_diagnostics(targets: pd.DataFrame) -> pd.DataFrame:
    diagnostics = targets.attrs.get(PORTFOLIO_DIAGNOSTICS_ATTR)
    if diagnostics is None:
        return pd.DataFrame(columns=DIAGNOSTIC_COLUMNS)
    return diagnostics.copy()


def build_candidate_pool_v2(scored_candidates: pd.DataFrame, constraints: PortfolioConstraints) -> pd.DataFrame:
    hard = apply_hard_filters(scored_candidates, constraints, score_column="trade_score_v2")
    if hard.empty:
        return hard.copy()
    mask = (
        (
            hard["absolute_rank_pct"].fillna(0.0) >= constraints.candidate_absolute_min_rank_pct
        )
        | (
            hard["active_rank_pct"].fillna(0.0) >= constraints.candidate_active_min_rank_pct
        )
    ) & (
        hard["risk_rank_pct"].fillna(1.0) <= constraints.candidate_risk_max_rank_pct
    ) & (
        hard["trade_score_v2"].fillna(-1.0) >= constraints.candidate_min_trade_score
    )
    return hard[mask].copy()


def build_core_pool_v2(scored_candidates: pd.DataFrame, constraints: PortfolioConstraints) -> pd.DataFrame:
    if scored_candidates.empty:
        return scored_candidates.copy()
    mask = (
        (scored_candidates["absolute_rank_pct"].fillna(0.0) >= constraints.core_absolute_min_rank_pct)
        & (scored_candidates["active_rank_pct"].fillna(0.0) >= constraints.core_active_min_rank_pct)
        & (scored_candidates["risk_rank_pct"].fillna(1.0) <= constraints.core_risk_max_rank_pct)
        & (scored_candidates["trade_score_v2"].fillna(-1.0) >= constraints.core_min_trade_score)
    )
    return scored_candidates[mask].copy()


def construct_portfolio_targets_v2(
    scored_candidates: pd.DataFrame,
    constraints: PortfolioConstraints,
    portfolio_id: str,
    current_holdings: pd.DataFrame | None = None,
    *,
    run_id: str | None = None,
    fold_id: str | None = None,
    score_version: str | None = None,
) -> pd.DataFrame:
    if "trade_date" in scored_candidates and scored_candidates["trade_date"].nunique(dropna=False) > 1:
        daily_targets = []
        daily_diagnostics = []
        holdings = current_holdings
        for _, day in scored_candidates.groupby("trade_date", sort=True):
            day_targets = _construct_portfolio_targets_v2_for_frame(
                day,
                constraints,
                portfolio_id,
                holdings,
                run_id=run_id,
                fold_id=fold_id,
                score_version=score_version,
            )
            daily_targets.append(day_targets)
            daily_diagnostics.append(get_portfolio_diagnostics(day_targets))
            holdings = _next_holdings_from_targets(day_targets)
        non_empty = [frame for frame in daily_targets if not frame.empty]
        concat_frames = [_without_attrs(frame) for frame in non_empty]
        out = pd.concat(concat_frames, ignore_index=True) if concat_frames else pd.DataFrame(columns=TARGET_COLUMNS)
        diagnostics = pd.concat(daily_diagnostics, ignore_index=True) if daily_diagnostics else pd.DataFrame(columns=DIAGNOSTIC_COLUMNS)
        return _attach_portfolio_diagnostics(out, diagnostics)
    return _construct_portfolio_targets_v2_for_frame(
        scored_candidates,
        constraints,
        portfolio_id,
        current_holdings,
        run_id=run_id,
        fold_id=fold_id,
        score_version=score_version,
    )


def _construct_portfolio_targets_v2_for_frame(
    scored_candidates: pd.DataFrame,
    constraints: PortfolioConstraints,
    portfolio_id: str,
    current_holdings: pd.DataFrame | None = None,
    *,
    run_id: str | None = None,
    fold_id: str | None = None,
    score_version: str | None = None,
) -> pd.DataFrame:
    raw = scored_candidates.copy()
    hard_filtered = apply_hard_filters(raw, constraints, score_column="trade_score_v2")
    core_pool = build_core_pool_v2(hard_filtered, constraints)
    candidate_pool = build_candidate_pool_v2(raw, constraints)
    retained, exit_rows, exit_decisions, retention_stats = _evaluate_current_holdings(
        raw,
        candidate_pool,
        constraints,
        current_holdings,
    )
    selected, blocked = _select_from_v2_pools(core_pool, candidate_pool, constraints, retained, current_holdings)
    selected = retained + selected + exit_rows
    now = datetime.now(timezone.utc).isoformat()
    targets = _targets_from_selected(selected, portfolio_id, now)
    diagnostics = pd.DataFrame(
        [
            _diagnostic_row(
                raw,
                hard_filtered,
                core_pool,
                candidate_pool,
                targets,
                constraints,
                portfolio_id,
                blocked,
                retained,
                exit_decisions,
                retention_stats,
                created_at=now,
                run_id=run_id,
                fold_id=fold_id,
                score_version=score_version,
            )
        ],
        columns=DIAGNOSTIC_COLUMNS,
    )
    return _attach_portfolio_diagnostics(targets, diagnostics)


def _select_from_v2_pools(
    core_pool: pd.DataFrame,
    candidate_pool: pd.DataFrame,
    constraints: PortfolioConstraints,
    retained_holdings: list[dict[str, object]],
    current_holdings: pd.DataFrame | None,
) -> tuple[list[dict[str, object]], dict[str, int]]:
    selected: list[dict[str, object]] = []
    selected_codes: set[str] = {str(row["code"]) for row in retained_holdings if row.get("code") is not None}
    industry_counts: dict[object, int] = {}
    unknown_industry_count = 0
    new_entries = 0
    held = _held_codes(current_holdings)
    position_limit = min(constraints.target_positions, constraints.hard_max_positions)
    if not held:
        position_limit = min(position_limit, constraints.max_initial_entries)
    new_entry_limit = constraints.max_initial_entries if not held else constraints.max_new_entries_per_day
    blocked = {
        "industry_limit_blocked_count": 0,
        "unknown_industry_limit_blocked_count": 0,
        "max_new_entries_blocked_count": 0,
    }

    core_codes = set(core_pool.get("code", pd.Series(dtype=object)).astype(str))
    ordered_pools = [
        ("core_pool", _sort_pool(core_pool)),
        (
            "candidate_pool",
            _sort_pool(candidate_pool[~candidate_pool["code"].astype(str).isin(core_codes)].copy())
            if "code" in candidate_pool
            else candidate_pool.copy(),
        ),
    ]
    for source, pool in ordered_pools:
        for row in pool.itertuples(index=False):
            if len(selected) + len(retained_holdings) >= position_limit:
                return selected, blocked
            code = str(getattr(row, "code"))
            if code in selected_codes:
                continue
            industry = getattr(row, "industry_code", None)
            industry_unknown = is_unknown_industry(industry)
            if industry_unknown and unknown_industry_count >= constraints.max_unknown_industry_names:
                blocked["unknown_industry_limit_blocked_count"] += 1
                continue
            if not industry_unknown and industry_counts.get(industry, 0) >= constraints.max_industry_names:
                blocked["industry_limit_blocked_count"] += 1
                continue
            if code not in held and new_entries >= new_entry_limit:
                blocked["max_new_entries_blocked_count"] += 1
                continue
            if code not in held:
                new_entries += 1
            if industry_unknown:
                unknown_industry_count += 1
            else:
                industry_counts[industry] = industry_counts.get(industry, 0) + 1
            row_dict = row._asdict()
            row_dict["_entry_reason"] = source
            row_dict["_signal_action"] = "buy"
            row_dict["_hold_reason"] = None
            row_dict["_exit_reason"] = None
            row_dict["_sell_blocked_reason"] = None
            selected.append(row_dict)
            selected_codes.add(code)
    return selected, blocked


def _evaluate_current_holdings(
    raw: pd.DataFrame,
    candidate_pool: pd.DataFrame,
    constraints: PortfolioConstraints,
    current_holdings: pd.DataFrame | None,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[SellDecision], dict[str, int | float]]:
    if current_holdings is None or current_holdings.empty or "code" not in current_holdings:
        return [], [], [], _empty_retention_stats()
    candidate_codes = _held_codes(candidate_pool)
    raw_by_code = _latest_rows_by_code(raw)
    trade_date = _first_value(raw, "trade_date")
    retained: list[dict[str, object]] = []
    exit_rows: list[dict[str, object]] = []
    decisions: list[SellDecision] = []
    stats = _empty_retention_stats()
    for holding_row in current_holdings.itertuples(index=False):
        holding_series = pd.Series(holding_row._asdict())
        code = str(holding_series["code"])
        latest_row = raw_by_code.get(code, holding_series.copy())
        latest_row = latest_row.copy()
        if "trade_date" not in latest_row or pd.isna(latest_row.get("trade_date")):
            latest_row["trade_date"] = trade_date
        latest_row["in_candidate_pool"] = code in candidate_codes
        holding = holding_state_from_row(holding_series)
        decision = evaluate_sell_decision(holding, latest_row, constraints.holding_policy)
        decisions.append(decision)
        if decision.should_sell:
            stats["sell_signal_count"] += 1
            if decision.blocked:
                stats["sell_blocked_count"] += 1
                retained.append(_retained_row(holding_series, latest_row, "sell_blocked", None, decision.reason, decision.reason))
            else:
                stats["sell_executed_count"] += 1
                _count_exit_reason(stats, decision.reason)
                exit_rows.append(_exit_row(holding_series, latest_row, decision.reason))
        else:
            stats[f"{decision.reason}_count"] = int(stats.get(f"{decision.reason}_count", 0)) + 1
            retained.append(_retained_row(holding_series, latest_row, "retained_holding", decision.reason, None, None))
    stats["retained_holdings_count"] = len(retained)
    holding_days = [
        int(row.get("holding_days", 0) or 0)
        for row in retained
    ]
    if holding_days:
        series = pd.Series(holding_days)
        stats["avg_holding_days_current"] = float(series.mean())
        stats["median_holding_days_current"] = float(series.median())
    return retained[: constraints.hard_max_positions], exit_rows, decisions, stats


def _retained_row(
    holding: pd.Series,
    latest: pd.Series,
    entry_reason: str,
    hold_reason: str | None,
    exit_reason: str | None,
    sell_blocked_reason: str | None,
) -> dict[str, object]:
    row = holding.to_dict()
    for column in [
        "trade_date",
        "industry_code",
        "industry_name",
        "trade_score_v2",
        "trade_score",
        "risk_rank_pct",
        "risk_prob",
        "absolute_rank_pct",
        "active_rank_pct",
    ]:
        if column in latest:
            row[column] = latest[column]
    row["_entry_reason"] = entry_reason
    row["_signal_action"] = "sell_blocked" if entry_reason == "sell_blocked" else "hold"
    row["_hold_reason"] = hold_reason
    row["_exit_reason"] = exit_reason
    row["_sell_blocked_reason"] = sell_blocked_reason
    row["latest_trade_score"] = latest.get("trade_score_v2", holding.get("latest_trade_score"))
    return row


def _exit_row(holding: pd.Series, latest: pd.Series, exit_reason: str) -> dict[str, object]:
    row = _retained_row(holding, latest, "sell_signal", None, exit_reason, None)
    row["_signal_action"] = "sell"
    row["target_weight"] = 0.0
    return row


def _targets_from_selected(selected: list[dict[str, object]], portfolio_id: str, generated_at: str) -> pd.DataFrame:
    if not selected:
        return pd.DataFrame(columns=TARGET_COLUMNS)
    out = pd.DataFrame(selected)
    out["portfolio_id"] = portfolio_id
    out["rank_n"] = range(1, len(out) + 1)
    out["target_weight"] = 0.0
    out["trade_score"] = out["trade_score_v2"] if "trade_score_v2" in out else out.get("trade_score", 0.0)
    out["entry_reason"] = out.pop("_entry_reason")
    out["signal_action"] = out.pop("_signal_action") if "_signal_action" in out else "buy"
    out["hold_reason"] = out.pop("_hold_reason") if "_hold_reason" in out else None
    out["exit_reason"] = out.pop("_exit_reason") if "_exit_reason" in out else None
    out["sell_blocked_reason"] = out.pop("_sell_blocked_reason") if "_sell_blocked_reason" in out else None
    out["generated_at"] = generated_at
    return out[[column for column in TARGET_COLUMNS if column in out.columns]]


def _diagnostic_row(
    raw: pd.DataFrame,
    hard_filtered: pd.DataFrame,
    core_pool: pd.DataFrame,
    candidate_pool: pd.DataFrame,
    targets: pd.DataFrame,
    constraints: PortfolioConstraints,
    portfolio_id: str,
    blocked: dict[str, int],
    retained: list[dict[str, object]],
    exit_decisions: list[SellDecision],
    retention_stats: dict[str, int | float],
    *,
    created_at: str,
    run_id: str | None,
    fold_id: str | None,
    score_version: str | None,
) -> dict[str, object]:
    selected_from_core = _entry_reason_count(targets, "core_pool")
    selected_from_candidate = _entry_reason_count(targets, "candidate_pool")
    final_selected_count = _active_target_count(targets)
    target_positions = max(int(constraints.target_positions), 1)
    cash_weight = max(0.0, 1.0 - min(final_selected_count, target_positions) / target_positions)
    return {
        "trade_date": _first_value(raw, "trade_date"),
        "run_id": run_id or _first_value(raw, "run_id") or "default_run",
        "fold_id": fold_id or _first_value(raw, "fold_id") or portfolio_id,
        "portfolio_id": portfolio_id,
        "score_version": score_version or _first_value(raw, "score_version") or "v2_three_model",
        "raw_candidate_count": len(raw),
        "hard_filter_pass_count": len(hard_filtered),
        "core_pool_size": len(core_pool),
        "candidate_pool_size": len(candidate_pool),
        "selected_from_core": selected_from_core,
        "selected_from_candidate": selected_from_candidate,
        "final_selected_count": final_selected_count,
        "low_adv_rejected_count": _low_adv_count(raw, constraints),
        "cannot_buy_rejected_count": _false_count(raw, "can_buy_next_open"),
        "st_rejected_count": _true_count(raw, "is_st"),
        "paused_rejected_count": _true_count(raw, "is_paused"),
        "bse_rejected_count": _bse_count(raw) if constraints.exclude_bse else 0,
        "low_trade_score_rejected_count": _low_score_count(raw, constraints),
        "high_risk_rejected_count": _high_risk_count(hard_filtered, constraints),
        "industry_limit_blocked_count": blocked["industry_limit_blocked_count"],
        "unknown_industry_limit_blocked_count": blocked["unknown_industry_limit_blocked_count"],
        "max_new_entries_blocked_count": blocked["max_new_entries_blocked_count"],
        "retained_holdings_count": retention_stats["retained_holdings_count"],
        "sell_signal_count": retention_stats["sell_signal_count"],
        "sell_executed_count": retention_stats["sell_executed_count"],
        "sell_blocked_count": retention_stats["sell_blocked_count"],
        "hold_due_to_min_days_count": retention_stats["hold_due_to_min_days_count"],
        "hold_due_to_score_ok_count": retention_stats["hold_due_to_score_ok_count"],
        "exit_due_to_score_count": retention_stats["exit_due_to_score_count"],
        "exit_due_to_risk_count": retention_stats["exit_due_to_risk_count"],
        "exit_due_to_time_count": retention_stats["exit_due_to_time_count"],
        "exit_due_to_not_candidate_count": retention_stats["exit_due_to_not_candidate_count"],
        "avg_holding_days_current": retention_stats["avg_holding_days_current"],
        "median_holding_days_current": retention_stats["median_holding_days_current"],
        "cash_weight": cash_weight,
        "created_at": created_at,
    }


def _attach_portfolio_diagnostics(targets: pd.DataFrame, diagnostics: pd.DataFrame) -> pd.DataFrame:
    out = targets.copy()
    out.attrs[PORTFOLIO_DIAGNOSTICS_ATTR] = diagnostics[[column for column in DIAGNOSTIC_COLUMNS if column in diagnostics.columns]].copy()
    return out


def _without_attrs(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out.attrs.clear()
    return out


def _sort_pool(pool: pd.DataFrame) -> pd.DataFrame:
    if pool.empty:
        return pool.copy()
    score_column = "trade_score_v2" if "trade_score_v2" in pool else "trade_score"
    return pool.sort_values([score_column, "code"], ascending=[False, True]).copy()


def _held_codes(current_holdings: pd.DataFrame | None) -> set[str]:
    if current_holdings is None or current_holdings.empty or "code" not in current_holdings:
        return set()
    return set(current_holdings["code"].dropna().astype(str))


def _next_holdings_from_targets(targets: pd.DataFrame) -> pd.DataFrame:
    if targets.empty or "code" not in targets:
        return pd.DataFrame(columns=["code"])
    holdings = targets.copy()
    if "target_weight" in holdings:
        holdings = holdings[pd.to_numeric(holdings["target_weight"], errors="coerce").fillna(0.0) > 0.0].copy()
    if holdings.empty:
        return pd.DataFrame(columns=["code"])
    if "entry_date" not in holdings:
        holdings["entry_date"] = holdings.get("trade_date")
    holdings["entry_date"] = holdings["entry_date"].fillna(holdings.get("trade_date"))
    if "entry_price" not in holdings:
        holdings["entry_price"] = 0.0
    if "shares" not in holdings:
        holdings["shares"] = 0.0
    if "holding_days" in holdings:
        holdings["holding_days"] = pd.to_numeric(holdings["holding_days"], errors="coerce").fillna(0).astype(int) + 1
    else:
        holdings["holding_days"] = 1
    if "calendar_days" in holdings:
        holdings["calendar_days"] = pd.to_numeric(holdings["calendar_days"], errors="coerce").fillna(0).astype(int) + 1
    else:
        holdings["calendar_days"] = holdings["holding_days"]
    if "entry_trade_score" not in holdings:
        holdings["entry_trade_score"] = holdings.get("trade_score")
    holdings["latest_trade_score"] = holdings.get("trade_score")
    keep = [
        "code",
        "entry_date",
        "entry_price",
        "shares",
        "holding_days",
        "calendar_days",
        "entry_trade_score",
        "latest_trade_score",
        "entry_reason",
        "industry_code",
        "industry_name",
    ]
    return holdings[[column for column in keep if column in holdings.columns]].copy()


def _first_value(frame: pd.DataFrame, column: str) -> object | None:
    if frame.empty or column not in frame:
        return None
    values = frame[column].dropna()
    if values.empty:
        return None
    return values.iloc[0]


def _entry_reason_count(targets: pd.DataFrame, reason: str) -> int:
    if targets.empty or "entry_reason" not in targets:
        return 0
    mask = targets["entry_reason"] == reason
    if "signal_action" in targets:
        mask &= targets["signal_action"].astype(str) != "sell"
    return int(mask.sum())


def _active_target_count(targets: pd.DataFrame) -> int:
    if targets.empty:
        return 0
    if "signal_action" not in targets:
        return len(targets)
    return int((targets["signal_action"].astype(str) != "sell").sum())


def _true_count(frame: pd.DataFrame, column: str) -> int:
    if column not in frame:
        return 0
    return int(frame[column].fillna(False).astype(bool).sum())


def _false_count(frame: pd.DataFrame, column: str) -> int:
    if column not in frame:
        return 0
    return int((~frame[column].fillna(False).astype(bool)).sum())


def _low_adv_count(frame: pd.DataFrame, constraints: PortfolioConstraints) -> int:
    if constraints.min_adv20_amount is None or "adv20_amount" not in frame:
        return 0
    return int((pd.to_numeric(frame["adv20_amount"], errors="coerce").fillna(0.0) < constraints.min_adv20_amount).sum())


def _low_score_count(frame: pd.DataFrame, constraints: PortfolioConstraints) -> int:
    if "trade_score_v2" not in frame:
        return 0
    return int((pd.to_numeric(frame["trade_score_v2"], errors="coerce").fillna(-1.0) < constraints.candidate_min_trade_score).sum())


def _high_risk_count(frame: pd.DataFrame, constraints: PortfolioConstraints) -> int:
    if "risk_rank_pct" not in frame:
        return 0
    return int((pd.to_numeric(frame["risk_rank_pct"], errors="coerce").fillna(1.0) > constraints.candidate_risk_max_rank_pct).sum())


def _bse_count(frame: pd.DataFrame) -> int:
    if "is_bse" in frame:
        return _true_count(frame, "is_bse")
    if "code" not in frame:
        return 0
    return int(frame["code"].map(detect_is_bse).sum())


def _latest_rows_by_code(frame: pd.DataFrame) -> dict[str, pd.Series]:
    if frame.empty or "code" not in frame:
        return {}
    return {
        str(row["code"]): row
        for _, row in frame.drop_duplicates("code", keep="last").iterrows()
    }


def _empty_retention_stats() -> dict[str, int | float]:
    return {
        "retained_holdings_count": 0,
        "sell_signal_count": 0,
        "sell_executed_count": 0,
        "sell_blocked_count": 0,
        "hold_due_to_min_days_count": 0,
        "hold_due_to_score_ok_count": 0,
        "exit_due_to_score_count": 0,
        "exit_due_to_risk_count": 0,
        "exit_due_to_time_count": 0,
        "exit_due_to_not_candidate_count": 0,
        "avg_holding_days_current": 0.0,
        "median_holding_days_current": 0.0,
    }


def _count_exit_reason(stats: dict[str, int | float], reason: str) -> None:
    if reason == "score_exit":
        stats["exit_due_to_score_count"] += 1
    elif reason == "risk_exit":
        stats["exit_due_to_risk_count"] += 1
    elif reason == "time_exit":
        stats["exit_due_to_time_count"] += 1
    elif reason == "not_candidate_after_target_days":
        stats["exit_due_to_not_candidate_count"] += 1
