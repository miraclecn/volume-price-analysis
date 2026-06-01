from __future__ import annotations

from vpa_structure_recognizer.batch_runner import (
    BatchPeriod,
    missing_month_batches,
    month_batches,
)


def test_month_batches_use_thirteen_month_warmup() -> None:
    batches = month_batches("2018-01-01", "2018-03-15")

    assert batches == [
        BatchPeriod("2016-12-01", "2018-01-01", "2018-01-31"),
        BatchPeriod("2017-01-01", "2018-02-01", "2018-02-28"),
        BatchPeriod("2017-02-01", "2018-03-01", "2018-03-15"),
    ]


def test_missing_month_batches_skip_completed_months() -> None:
    batches = month_batches("2018-01-01", "2018-04-30")

    missing = missing_month_batches(batches, {"2018-01", "2018-03"})

    assert [batch.month_key for batch in missing] == ["2018-02", "2018-04"]
