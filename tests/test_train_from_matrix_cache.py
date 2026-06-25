from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

from ml_stock_selector.feature_matrix import FeatureSchema, save_feature_schema
from ml_stock_selector.matrix_cache import FoldMatrixCache
from ml_stock_selector.models.fold_cache_training import (
    _mean_group_rank_ic,
    train_alpha_risk_models_from_fold_cache,
    train_three_models_from_fold_cache,
)


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

    artifacts = train_three_models_from_fold_cache(
        cache,
        SimpleNamespace(model={"lightgbm_runtime": {"n_estimators": 7, "num_leaves": 11}}),
        tmp_path / "artifacts",
    )

    assert len(artifacts.model_ids) == 3
    assert artifacts.absolute.artifact_uri.exists()
    assert artifacts.active.artifact_uri.exists()
    assert artifacts.risk.artifact_uri.exists()
    assert artifacts.absolute.feature_schema_uri == cache.feature_schema_path
    params_path = artifacts.absolute.artifact_dir / f"{artifacts.absolute.model_id}.params.json"
    params = json.loads(params_path.read_text(encoding="utf-8"))
    assert params["n_estimators"] == 7
    assert params["num_leaves"] == 11


def test_train_alpha_risk_models_from_fold_cache_skips_active_ranker(tmp_path):
    cache = FoldMatrixCache.from_paths("run", "wf", tmp_path / "cache")
    cache.cache_dir.mkdir(parents=True)
    x = sparse.csr_matrix(np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32))
    sparse.save_npz(cache.x_train_path, x)
    sparse.save_npz(cache.x_valid_path, x)
    np.save(cache.y_abs_train_path, np.array([0, 1], dtype=np.float32))
    np.save(cache.y_abs_valid_path, np.array([0, 1], dtype=np.float32))
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

    artifacts = train_alpha_risk_models_from_fold_cache(
        cache,
        SimpleNamespace(model={"lightgbm_runtime": {"n_estimators": 7, "num_leaves": 11}}),
        tmp_path / "artifacts",
    )

    assert len(artifacts.model_ids) == 2
    assert artifacts.absolute.artifact_uri.exists()
    assert artifacts.risk.artifact_uri.exists()
    assert not hasattr(artifacts, "active")
    assert artifacts.absolute.label_name == "absolute_label"
    assert artifacts.risk.label_name == "risk_label"


def test_train_alpha_risk_models_from_fold_cache_uses_future_max_gain_eval_metric(tmp_path):
    pytest.importorskip("lightgbm")
    cache = FoldMatrixCache.from_paths("run", "wf", tmp_path / "cache")
    cache.cache_dir.mkdir(parents=True)
    x_train = sparse.csr_matrix(
        np.array(
            [
                [0.10, 0.00],
                [0.20, 0.10],
                [0.30, 0.20],
                [0.40, 0.30],
                [0.00, 0.10],
                [0.10, 0.20],
                [0.20, 0.30],
                [0.30, 0.40],
            ],
            dtype=np.float32,
        )
    )
    x_valid = sparse.csr_matrix(
        np.array(
            [
                [0.15, 0.00],
                [0.25, 0.10],
                [0.35, 0.20],
                [0.45, 0.30],
                [0.05, 0.10],
                [0.15, 0.20],
                [0.25, 0.30],
                [0.35, 0.40],
            ],
            dtype=np.float32,
        )
    )
    sparse.save_npz(cache.x_train_path, x_train)
    sparse.save_npz(cache.x_valid_path, x_valid)
    np.save(cache.y_abs_train_path, np.array([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.float32))
    np.save(cache.y_abs_valid_path, np.array([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.float32))
    np.save(cache.y_risk_train_path, np.array([0, 0, 1, 1, 0, 1, 0, 1], dtype=np.float32))
    np.save(cache.y_risk_valid_path, np.array([0, 0, 1, 1, 0, 1, 0, 1], dtype=np.float32))
    np.save(cache.group_train_path, np.array([4, 4], dtype=np.int32))
    np.save(cache.group_valid_path, np.array([4, 4], dtype=np.int32))
    np.save(cache.cache_dir / "y_alpha_eval_valid_future_max_gain.npy", np.array([0.01, 0.05, 0.03, 0.07, 0.02, 0.06, 0.04, 0.08], dtype=np.float32))
    save_feature_schema(
        FeatureSchema("vpa_d_sequence", ["ret_1d", "range_pct"], [], ["ret_1d", "range_pct"], {}, {}, "v2"),
        cache.feature_schema_path,
    )
    cache.manifest_path.write_text(
        json.dumps(
            {
                "feature_set_id": "vpa_d_sequence",
                "label_base": "from_next_open",
                "horizon_d": 5,
                "alpha_eval_metric": "future_max_gain_rank_ic",
                "alpha_eval_valid_path": "y_alpha_eval_valid_future_max_gain.npy",
            }
        ),
        encoding="utf-8",
    )

    artifacts = train_alpha_risk_models_from_fold_cache(
        cache,
        SimpleNamespace(
            model={
                "lightgbm_runtime": {
                    "n_estimators": 10,
                    "early_stopping_rounds": 2,
                    "num_leaves": 5,
                    "min_data_in_leaf": 1,
                    "random_state": 7,
                    "force_col_wise": True,
                }
            }
        ),
        tmp_path / "artifacts",
    )

    assert "best_score_valid_0_future_max_gain_rank_ic" in artifacts.absolute.metrics
    assert "best_score_valid_0_ndcg@10" not in artifacts.absolute.metrics
    assert "best_score_valid_0_ndcg@15" not in artifacts.absolute.metrics


def test_group_rank_ic_tracks_monotonic_order_by_group():
    target = np.array([0.01, 0.03, 0.02, 0.00, 0.05, 0.02], dtype=np.float32)
    aligned = np.array([1.0, 3.0, 2.0, 1.0, 3.0, 2.0], dtype=np.float32)
    reversed_pred = np.array([3.0, 1.0, 2.0, 3.0, 1.0, 2.0], dtype=np.float32)

    assert _mean_group_rank_ic(target, aligned, [3, 3]) > 0.99
    assert _mean_group_rank_ic(target, reversed_pred, [3, 3]) < -0.99
