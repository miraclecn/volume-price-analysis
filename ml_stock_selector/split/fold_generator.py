from __future__ import annotations

from typing import Literal


ExperimentName = Literal["expanding_gap", "expanding_nogap", "rolling5_gap", "rolling5_nogap"]


def generate_expanding_folds(
    *,
    train_start: str = "2015-01-05",
    first_test_year: int = 2020,
    last_test_year: int = 2026,
    gap_years: int = 1,
    last_test_end: str | None = None,
    embargo_days: int = 0,
) -> list[dict[str, object]]:
    folds = []
    for test_year in range(first_test_year, last_test_year + 1):
        train_end_year = test_year - gap_years - 1
        valid_year = test_year - 1
        valid_start = f"{valid_year}-01-01" if gap_years else f"{test_year}-01-01"
        valid_end = f"{valid_year}-12-31" if gap_years else f"{train_end_year}-12-31"
        folds.append(
            _fold(
                test_year,
                train_start=train_start,
                train_end=f"{train_end_year}-12-31",
                valid_start=valid_start,
                valid_end=valid_end,
                test_end=_test_end(test_year, last_test_year, last_test_end),
                gap_type="one_year_gap" if gap_years else "no_gap",
                embargo_days=embargo_days,
            )
        )
    return folds


def generate_rolling_folds(
    *,
    train_start: str = "2015-01-05",
    first_test_year: int = 2020,
    last_test_year: int = 2026,
    train_years: int = 5,
    gap_years: int = 1,
    last_test_end: str | None = None,
    embargo_days: int = 0,
) -> list[dict[str, object]]:
    folds = []
    base_start_year = int(train_start[:4])
    for test_year in range(first_test_year, last_test_year + 1):
        train_end_year = test_year - gap_years - 1
        rolling_start_year = train_end_year - train_years + 1
        start = train_start if rolling_start_year <= base_start_year else f"{rolling_start_year}-01-01"
        valid_year = test_year - 1
        valid_start = f"{valid_year}-01-01" if gap_years else f"{test_year}-01-01"
        valid_end = f"{valid_year}-12-31" if gap_years else f"{train_end_year}-12-31"
        folds.append(
            _fold(
                test_year,
                train_start=start,
                train_end=f"{train_end_year}-12-31",
                valid_start=valid_start,
                valid_end=valid_end,
                test_end=_test_end(test_year, last_test_year, last_test_end),
                gap_type="rolling5_gap" if gap_years else "rolling5_nogap",
                embargo_days=embargo_days,
            )
        )
    return folds


def generate_experiment_folds(
    experiment_name: ExperimentName | str,
    *,
    train_start: str = "2015-01-05",
    first_test_year: int = 2020,
    last_test_year: int = 2026,
    last_test_end: str | None = None,
    embargo_days: int = 0,
) -> list[dict[str, object]]:
    if experiment_name == "expanding_gap":
        return generate_expanding_folds(
            train_start=train_start,
            first_test_year=first_test_year,
            last_test_year=last_test_year,
            gap_years=1,
            last_test_end=last_test_end,
            embargo_days=embargo_days,
        )
    if experiment_name == "expanding_nogap":
        return generate_expanding_folds(
            train_start=train_start,
            first_test_year=first_test_year,
            last_test_year=last_test_year,
            gap_years=0,
            last_test_end=last_test_end,
            embargo_days=embargo_days,
        )
    if experiment_name == "rolling5_gap":
        return generate_rolling_folds(
            train_start=train_start,
            first_test_year=first_test_year,
            last_test_year=last_test_year,
            train_years=5,
            gap_years=1,
            last_test_end=last_test_end,
            embargo_days=embargo_days,
        )
    if experiment_name == "rolling5_nogap":
        return generate_rolling_folds(
            train_start=train_start,
            first_test_year=first_test_year,
            last_test_year=last_test_year,
            train_years=5,
            gap_years=0,
            last_test_end=last_test_end,
            embargo_days=embargo_days,
        )
    raise ValueError(f"Unknown experiment_name: {experiment_name}")


def _fold(
    test_year: int,
    *,
    train_start: str,
    train_end: str,
    valid_start: str,
    valid_end: str,
    test_end: str,
    gap_type: str,
    embargo_days: int,
) -> dict[str, object]:
    return {
        "fold_id": f"wf_{test_year}",
        "train_start": train_start,
        "train_end": train_end,
        "valid_start": valid_start,
        "valid_end": valid_end,
        "test_start": f"{test_year}-01-01",
        "test_end": test_end,
        "gap_type": gap_type,
        "embargo_days": int(embargo_days),
    }


def _test_end(test_year: int, last_test_year: int, last_test_end: str | None) -> str:
    if last_test_end is not None and test_year == last_test_year:
        return last_test_end
    return f"{test_year}-12-31"
