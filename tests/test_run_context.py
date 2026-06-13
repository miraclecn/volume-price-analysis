from __future__ import annotations

import json

from ml_stock_selector.runtime.run_context import create_run_context, register_run_context, register_run_fold
from ml_stock_selector.storage import init_ml_db


def test_run_context_writes_manifest_and_database_records(tmp_path):
    config_path = tmp_path / "config.toml"
    config_path.write_text("[data]\nalpha_data_db = 'research.duckdb'\n", encoding="utf-8")
    con = init_ml_db(tmp_path / "ml.duckdb")

    context = create_run_context(
        run_type="walkforward",
        run_id="run_ctx",
        experiment_name="expanding_gap",
        config_path=config_path,
        artifact_root=tmp_path / "runs",
        alpha_data_db="/data/research.duckdb",
        ml_db=str(tmp_path / "ml.duckdb"),
        feature_set_id="vpa_d_sequence",
        feature_store_version="v2",
        label_version="from_next_open_h5",
        score_version="v2_three_model",
    )
    register_run_context(con, context)
    fold_dir = register_run_fold(
        con,
        context,
        {
            "fold_id": "wf_2020",
            "train_start": "2015-01-01",
            "train_end": "2019-12-31",
            "valid_start": "2020-01-01",
            "valid_end": "2020-06-30",
            "test_start": "2020-07-01",
            "test_end": "2020-12-31",
            "gap_type": "one_year_gap",
            "embargo_days": 0,
        },
    )

    run_manifest = json.loads((context.artifact_root / "run_manifest.json").read_text(encoding="utf-8"))
    fold_manifest = json.loads((fold_dir / "fold_manifest.json").read_text(encoding="utf-8"))
    row = con.execute(
        "select run_type, config_hash, artifact_root from ml_runs where run_id = 'run_ctx'"
    ).fetchone()
    fold_row = con.execute(
        "select test_start, artifact_dir from ml_run_folds where run_id = 'run_ctx' and fold_id = 'wf_2020'"
    ).fetchone()
    con.close()

    assert context.artifact_root == tmp_path / "runs" / "run_ctx"
    assert (context.artifact_root / "config_snapshot.toml").exists()
    assert (context.artifact_root / "config_hash.txt").read_text(encoding="utf-8").strip() == context.config_hash
    assert (context.artifact_root / "git_commit.txt").exists()
    assert run_manifest["run_id"] == "run_ctx"
    assert run_manifest["feature_set_id"] == "vpa_d_sequence"
    assert fold_manifest["run_id"] == "run_ctx"
    assert fold_manifest["fold_id"] == "wf_2020"
    assert row == ("walkforward", context.config_hash, str(context.artifact_root))
    assert fold_row == ("2020-07-01", str(fold_dir))
