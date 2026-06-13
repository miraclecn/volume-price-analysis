from __future__ import annotations

import json

from ml_stock_selector.matrix_cache import (
    FoldMatrixCache,
    is_fold_manifest_complete,
    read_fold_manifest,
    update_fold_manifest_status,
)


def test_fold_manifest_status_supports_resume_stage_checks(tmp_path):
    cache = FoldMatrixCache.from_paths("run", "wf_test", tmp_path / "cache")
    cache.cache_dir.mkdir(parents=True)
    for path in [
        cache.x_train_path,
        cache.x_valid_path,
        cache.x_test_path,
        cache.y_abs_train_path,
        cache.y_abs_valid_path,
        cache.y_active_train_path,
        cache.y_active_valid_path,
        cache.y_risk_train_path,
        cache.y_risk_valid_path,
        cache.group_train_path,
        cache.group_valid_path,
        cache.metadata_train_path,
        cache.metadata_valid_path,
        cache.metadata_test_path,
        cache.feature_schema_path,
    ]:
        path.write_bytes(b"artifact")
    cache.manifest_path.write_text(
        json.dumps({"run_id": "run", "fold_id": "wf_test", "status": "matrix_built"}),
        encoding="utf-8",
    )

    assert is_fold_manifest_complete(cache, "matrix_built")
    assert not is_fold_manifest_complete(cache, "models_trained")

    update_fold_manifest_status(
        cache.manifest_path,
        "models_trained",
        artifacts={
            "absolute": {"model_id": "abs", "artifact_uri": "abs.pkl", "feature_schema_uri": "schema.json"},
            "active": {"model_id": "act", "artifact_uri": "act.pkl", "feature_schema_uri": "schema.json"},
            "risk": {"model_id": "risk", "artifact_uri": "risk.pkl", "feature_schema_uri": "schema.json"},
        },
    )

    manifest = read_fold_manifest(cache.manifest_path)
    assert manifest["status"] == "models_trained"
    assert manifest["artifacts"]["absolute"]["model_id"] == "abs"
    assert is_fold_manifest_complete(cache, "models_trained")
