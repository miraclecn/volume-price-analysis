from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class DateRange:
    start: str
    end: str


@dataclass(frozen=True)
class TrainValidTestSplit:
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame


def make_walk_forward_split(
    samples: pd.DataFrame,
    train_range: DateRange,
    valid_range: DateRange,
    test_range: DateRange,
    embargo_days: int,
) -> TrainValidTestSplit:
    frame = samples.copy()
    dates = pd.to_datetime(frame["trade_date"])
    valid_start = pd.to_datetime(valid_range.start)
    test_start = pd.to_datetime(test_range.start)
    train = frame[
        (frame["trade_date"] >= train_range.start)
        & (frame["trade_date"] <= train_range.end)
        & (dates < valid_start - pd.Timedelta(days=embargo_days))
    ]
    valid = frame[
        (frame["trade_date"] >= valid_range.start)
        & (frame["trade_date"] <= valid_range.end)
        & (dates <= test_start - pd.Timedelta(days=embargo_days))
    ]
    test = frame[(frame["trade_date"] >= test_range.start) & (frame["trade_date"] <= test_range.end)]
    return TrainValidTestSplit(train.reset_index(drop=True), valid.reset_index(drop=True), test.reset_index(drop=True))


def build_lgbm_group(frame: pd.DataFrame) -> list[int]:
    return [int(value) for value in frame.sort_values(["trade_date", "code"]).groupby("trade_date", sort=True).size().tolist()]
