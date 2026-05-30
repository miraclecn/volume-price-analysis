from __future__ import annotations

import pandas as pd


UNKNOWN_INDUSTRY_CODES = {"UNKNOWN", "UNKNOWN_SECTOR", ""}

STATE_SCORES = {
    "HEALTHY_UPTREND": 95.0,
    "POSSIBLE_ACCUMULATION": 75.0,
    "LOW_LEVEL_SUPPORT": 65.0,
    "DECLINE_EXHAUSTION": 55.0,
    "BREAKOUT_ATTEMPT": 50.0,
    "UNCLEAR": 50.0,
    "HIGH_LEVEL_SUPPLY": 35.0,
    "POSSIBLE_DISTRIBUTION": 25.0,
    "BREAKDOWN": 20.0,
}


def rank_top_down(
    states: pd.DataFrame,
    stock_sector_map: dict[str, str],
    weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    weights = weights or {
        "market": 0.25,
        "sector": 0.30,
        "stock": 0.35,
        "resonance": 0.10,
    }
    ranked = states.copy()
    for column in [
        "market_score",
        "sector_score",
        "self_score",
        "relative_strength_score",
        "resonance_score",
        "final_rating",
    ]:
        if column not in ranked.columns:
            ranked[column] = None

    output_rows = []
    for date, group in ranked.groupby("date", sort=True):
        market_score = _market_score(group)
        sector_scores = {
            row.scope_id: _state_score(row.final_state)
            for row in group[group["scope_type"] == "sector"].itertuples(index=False)
            if not _is_unknown_industry(row.scope_id)
        }
        for row in group.itertuples(index=False):
            row_dict = row._asdict()
            self_score = _state_score(row.final_state)
            if row.scope_type == "market":
                _apply_scores(row_dict, market_score, None, self_score, None, 0.0, self_score)
            elif row.scope_type == "sector":
                sector_score = sector_scores.get(row.scope_id, self_score)
                _apply_scores(
                    row_dict,
                    market_score,
                    sector_score,
                    self_score,
                    sector_score - market_score,
                    100.0 if market_score > 60 and sector_score > 60 else 0.0,
                    sector_score,
                )
            else:
                sector_id = stock_sector_map.get(row.scope_id)
                industry_unknown = _is_unknown_industry(sector_id)
                sector_score = 50.0 if industry_unknown else sector_scores.get(sector_id, 50.0)
                if industry_unknown:
                    row_dict["risk_flags"] = _append_flag(
                        row_dict.get("risk_flags"), "industry_unknown"
                    )
                resonance = 100.0 if market_score > 60 and sector_score > 60 and self_score > 60 else 0.0
                final_score = (
                    market_score * weights["market"]
                    + sector_score * weights["sector"]
                    + self_score * weights["stock"]
                    + resonance * weights["resonance"]
                )
                rating = _rating(final_score)
                if market_score < 40:
                    rating = _downgrade(rating)
                if sector_score < 40:
                    rating = _downgrade(rating)
                    if self_score > sector_score:
                        row_dict["risk_flags"] = _append_flag(
                            row_dict.get("risk_flags"), "逆势个股，仅观察"
                        )
                _apply_scores(
                    row_dict,
                    market_score,
                    sector_score,
                    self_score,
                    self_score - sector_score,
                    resonance,
                    final_score,
                    rating,
                )
            output_rows.append(row_dict)

    return pd.DataFrame(output_rows)


def _market_score(group: pd.DataFrame) -> float:
    market = group[group["scope_type"] == "market"]
    if market.empty:
        return 50.0
    return _state_score(market.iloc[0]["final_state"])


def _state_score(final_state: str) -> float:
    return STATE_SCORES.get(final_state, 50.0)


def _is_unknown_industry(value: object) -> bool:
    if value is None or pd.isna(value):
        return True
    return str(value).strip().upper() in UNKNOWN_INDUSTRY_CODES


def _apply_scores(
    row: dict[str, object],
    market_score: float | None,
    sector_score: float | None,
    self_score: float,
    relative_strength_score: float | None,
    resonance_score: float,
    final_score: float,
    final_rating: str | None = None,
) -> None:
    row["market_score"] = market_score
    row["sector_score"] = sector_score
    row["self_score"] = self_score
    row["relative_strength_score"] = relative_strength_score
    row["resonance_score"] = resonance_score
    row["final_rating"] = final_rating or _rating(final_score)


def _rating(score: float) -> str:
    if score >= 80:
        return "A"
    if score >= 65:
        return "B"
    if score >= 50:
        return "C"
    if score >= 35:
        return "D"
    return "E"


def _downgrade(rating: str) -> str:
    order = ["A", "B", "C", "D", "E"]
    index = order.index(rating)
    return order[min(index + 1, len(order) - 1)]


def _append_flag(existing: object, flag: str) -> str:
    if existing is None or existing == "":
        return flag
    return f"{existing}; {flag}"
