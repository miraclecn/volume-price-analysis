from __future__ import annotations

from scripts.run_ml_walkforward import build_arg_parser


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


def test_walkforward_cli_requires_explicit_legacy_json_path_flag():
    from scripts.run_ml_walkforward import build_arg_parser

    parser = build_arg_parser()
    args = parser.parse_args(["--allow-legacy-json-path", "--force"])

    assert args.allow_legacy_json_path is True
    assert args.force is True
