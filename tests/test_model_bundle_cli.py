from __future__ import annotations

import pytest

from scripts.activate_model_bundle import build_arg_parser as build_activate_parser
from scripts.activate_model_bundle import main as activate_main
from scripts.train_production_bundle import build_arg_parser as build_train_parser


def test_train_production_bundle_cli_requires_run_id_and_defaults_to_candidate_production_bundle():
    args = build_train_parser().parse_args(["--run-id", "prod_core_20260613"])

    assert args.run_id == "prod_core_20260613"
    assert args.bundle_id is None
    assert args.bundle_role == "production"
    assert args.score_version == "v2_three_model"


def test_activate_model_bundle_cli_requires_confirm(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["activate_model_bundle.py", "--bundle-id", "prod_core_20260613", "--ml-db", ":memory:"],
    )

    with pytest.raises(SystemExit, match="--confirm is required"):
        activate_main()


def test_activate_model_bundle_cli_accepts_explicit_ml_db():
    args = build_activate_parser().parse_args(
        ["--bundle-id", "prod_core_20260613", "--ml-db", "outputs/ml/ml.duckdb", "--confirm"]
    )

    assert args.bundle_id == "prod_core_20260613"
    assert args.ml_db == "outputs/ml/ml.duckdb"
    assert args.confirm is True
