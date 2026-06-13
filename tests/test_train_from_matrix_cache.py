from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
from scipy import sparse

from ml_stock_selector.feature_matrix import FeatureSchema, save_feature_schema
from ml_stock_selector.matrix_cache import FoldMatrixCache
from ml_stock_selector.models.fold_cache_training import train_three_models_from_fold_cache


def test_train_three_models_from_fold_cache_creates_three_inactive_ready_artifacts(tmp_path):
    cache = FoldMatrixCache.from_paths("run", "wf", tmp_path / "cache")
    cache.cache_dir.mkdir(parents=True)
    x = sparse.csr_matrix(np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32))
    sparse.save_npz(cache.x_train_path, x)
    sparse.save_npz(cache.x_valid_path, x)
    sparse.save_npz(cache.x_test_path, x)
    np.save(cache.y_abs_train_path, np.array([0, 1], dtype=np.float32))
    np.save(cache.y_abs_valid_path, np.array([0, 1], dtype=np.float32))
    np.save(cache.y_active_train_path, np.array([1, 0], dtype=np.float32))
    np.save(cache.y_active_valid_path, np.array([1, 0], dtype=np.float32))
    np.save(cache.y_risk_train_path, np.array([0, 1], dtype=np.float32))
    np.save(cache.y_risk_valid_path, np.array([0, 1], dtype=np.float32))
    np.save(cache.group_train_path, np.array([2], dtype=np.int32))
    np.save(cache.group_valid_path, np.array([2], dtype=np.int32))
    save_feature_schema(
        FeatureSchema("vpa_d_sequence", ["ret_1d", "range_pct"], [], ["ret_1d", "range_pct"], {}, {}, "v2"),
        cache.feature_schema_path,
    )
    cache.manifest_path.write_text(
        json.dumps({"feature_set_id": "vpa_d_sequence", "label_base": "from_next_open", "horizon_d": 5}),
        encoding="utf-8",
    )

    artifacts = train_three_models_from_fold_cache(cache, SimpleNamespace(model={}), tmp_path / "artifacts")

    assert len(artifacts.model_ids) == 3
    assert artifacts.absolute.artifact_uri.exists()
    assert artifacts.active.artifact_uri.exists()
    assert artifacts.risk.artifact_uri.exists()
    assert artifacts.absolute.feature_schema_uri == cache.feature_schema_path
