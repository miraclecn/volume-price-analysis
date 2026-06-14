from __future__ import annotations

from ml_stock_selector.split.fold_generator import generate_experiment_folds, generate_expanding_folds, generate_rolling_folds


def test_generate_expanding_gap_folds_use_prior_year_validation_and_test_year():
    folds = generate_expanding_folds(
        train_start="2015-01-05",
        first_test_year=2020,
        last_test_year=2022,
        gap_years=1,
    )

    assert folds == [
        {
            "fold_id": "wf_2020",
            "train_start": "2015-01-05",
            "train_end": "2018-12-31",
            "valid_start": "2019-01-01",
            "valid_end": "2019-12-31",
            "test_start": "2020-01-01",
            "test_end": "2020-12-31",
            "gap_type": "one_year_gap",
            "embargo_days": 0,
        },
        {
            "fold_id": "wf_2021",
            "train_start": "2015-01-05",
            "train_end": "2019-12-31",
            "valid_start": "2020-01-01",
            "valid_end": "2020-12-31",
            "test_start": "2021-01-01",
            "test_end": "2021-12-31",
            "gap_type": "one_year_gap",
            "embargo_days": 0,
        },
        {
            "fold_id": "wf_2022",
            "train_start": "2015-01-05",
            "train_end": "2020-12-31",
            "valid_start": "2021-01-01",
            "valid_end": "2021-12-31",
            "test_start": "2022-01-01",
            "test_end": "2022-12-31",
            "gap_type": "one_year_gap",
            "embargo_days": 0,
        },
    ]


def test_generate_expanding_nogap_folds_have_empty_validation_window():
    folds = generate_expanding_folds(
        train_start="2015-01-05",
        first_test_year=2020,
        last_test_year=2021,
        gap_years=0,
    )

    assert folds[0]["train_end"] == "2019-12-31"
    assert folds[0]["valid_start"] == "2020-01-01"
    assert folds[0]["valid_end"] == "2019-12-31"
    assert folds[0]["gap_type"] == "no_gap"


def test_generate_rolling5_gap_folds_slide_train_start_after_initial_window():
    folds = generate_rolling_folds(
        train_start="2015-01-05",
        first_test_year=2020,
        last_test_year=2023,
        train_years=5,
        gap_years=1,
    )

    assert folds[0]["train_start"] == "2015-01-05"
    assert folds[1]["train_start"] == "2015-01-05"
    assert folds[2]["train_start"] == "2016-01-01"
    assert folds[3]["train_start"] == "2017-01-01"
    assert {fold["gap_type"] for fold in folds} == {"rolling5_gap"}


def test_generate_experiment_folds_supports_four_phase7_training_modes():
    assert generate_experiment_folds("expanding_gap", last_test_year=2020)[0]["gap_type"] == "one_year_gap"
    assert generate_experiment_folds("expanding_nogap", last_test_year=2020)[0]["gap_type"] == "no_gap"
    assert generate_experiment_folds("rolling5_gap", last_test_year=2020)[0]["gap_type"] == "rolling5_gap"
    assert generate_experiment_folds("rolling5_nogap", last_test_year=2020)[0]["gap_type"] == "rolling5_nogap"
