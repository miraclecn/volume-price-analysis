from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StrategySleeve:
    sleeve: str
    strategy_id: str
    score_version: str
    bundle_id: str | None
    experiment_family: str | None = None
    gap_type: str | None = None
    model_roles: tuple[str, ...] = ()
    enabled: bool = True


def default_phase9_sleeves(
    *,
    core_bundle_id: str | None = None,
    aggressive_bundle_id: str | None = None,
    fixed_horizon_bundle_id: str | None = None,
) -> list[StrategySleeve]:
    return [
        StrategySleeve(
            sleeve="core",
            strategy_id="holding_aware_v2",
            score_version="v2_absolute_risk_filter",
            bundle_id=core_bundle_id,
            experiment_family="expanding_gap",
            gap_type="one_year_gap",
            model_roles=("absolute", "risk"),
        ),
        StrategySleeve(
            sleeve="aggressive",
            strategy_id="holding_aware_v2",
            score_version="v2_three_model",
            bundle_id=aggressive_bundle_id,
            experiment_family="expanding_nogap_or_rolling5",
            gap_type="no_gap_or_short_gap",
            model_roles=("absolute", "active", "risk"),
        ),
        StrategySleeve(
            sleeve="fixed_horizon",
            strategy_id="abs_ranker_fixed_5d_risk_filter_v1",
            score_version="abs_ranker_fixed_5d_risk_filter_v1",
            bundle_id=fixed_horizon_bundle_id,
            experiment_family="fixed_horizon",
            gap_type=None,
            model_roles=("absolute", "risk"),
        ),
        StrategySleeve(
            sleeve="cash",
            strategy_id="cash_reserve",
            score_version="cash",
            bundle_id=None,
            experiment_family=None,
            gap_type=None,
            model_roles=(),
        ),
    ]

