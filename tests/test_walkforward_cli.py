from __future__ import annotations

import pytest

from ml_stock_selector.config import load_ml_config
from scripts.run_ml_walkforward import build_arg_parser
from scripts.run_ml_walkforward import resolve_selected_folds, validate_stage_range


def test_walkforward_cli_accepts_p0_feature_store_arguments():
    args = build_arg_parser().parse_args(
        [
            "--config",
            "config/ml_walkforward.toml",
            "--ml-db",
            "outputs/ml/ml.duckdb",
            "--run-id",
            "wf_three_model_v2_parquet_001",
            "--fold-id",
            "wf_2020",
            "--feature-store-dir",
            "outputs/ml/feature_store",
            "--feature-store-version",
            "v2_pv_only_001",
            "--use-feature-store",
            "true",
            "--matrix-cache-dir",
            "outputs/ml/cache/folds",
            "--feature-set-id",
            "vpa_d_sequence",
            "--horizon-d",
            "5",
            "--label-base",
            "from_next_open",
            "--score-version",
            "v2_three_model",
        ]
    )

    assert args.fold_id == "wf_2020"
    assert args.use_feature_store is True
    assert args.feature_store_version == "v2_pv_only_001"
    assert args.force is False


def test_walkforward_cli_accepts_phase7_stage_and_dry_run_arguments():
    args = build_arg_parser().parse_args(
        [
            "--config",
            "config/experiments/expanding_gap.toml",
            "--run-id",
            "20260613_expanding_gap_v1",
            "--experiment-name",
            "expanding_gap",
            "--from-stage",
            "matrix",
            "--to-stage",
            "backtest",
            "--dry-run",
            "--force",
        ]
    )

    assert args.experiment_name == "expanding_gap"
    assert args.from_stage == "matrix"
    assert args.to_stage == "backtest"
    assert args.dry_run is True
    assert args.force is True


def test_walkforward_stage_range_rejects_reverse_order():
    with pytest.raises(ValueError, match="from-stage"):
        validate_stage_range("backtest", "matrix")


def test_resolve_selected_folds_can_generate_phase7_experiment_folds():
    args = build_arg_parser().parse_args(
        [
            "--experiment-name",
            "rolling5_nogap",
            "--generated-first-test-year",
            "2020",
            "--generated-last-test-year",
            "2021",
        ]
    )

    folds = resolve_selected_folds(args, config_folds=[])

    assert [fold["fold_id"] for fold in folds] == ["wf_2020", "wf_2021"]
    assert {fold["gap_type"] for fold in folds} == {"rolling5_nogap"}


def test_phase7_experiment_configs_are_loadable():
    for name in ["expanding_gap", "expanding_nogap", "rolling5_gap", "rolling5_nogap"]:
        config = load_ml_config(f"config/experiments/{name}.toml")
        assert config.features["feature_set_id"] == "vpa_d_sequence"
        assert config.split["folds"]


def test_walkforward_cli_requires_explicit_legacy_json_path_flag():
    from scripts.run_ml_walkforward import build_arg_parser

    parser = build_arg_parser()
    args = parser.parse_args(["--allow-legacy-json-path", "--force"])

    assert args.allow_legacy_json_path is True
    assert args.force is True
