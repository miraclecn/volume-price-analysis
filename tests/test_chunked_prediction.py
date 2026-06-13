from __future__ import annotations

import json

import numpy as np
from scipy import sparse

from ml_stock_selector.feature_matrix import FeatureSchema, save_feature_schema
from ml_stock_selector.matrix_cache import FoldMatrixCache
from ml_stock_selector.models.alpha_ranker import LinearFallbackModel
from ml_stock_selector.models.risk_model import LogisticFallbackModel
from ml_stock_selector.prediction import write_chunked_fold_predictions
from ml_stock_selector.storage import init_ml_db


def test_chunked_prediction_writes_raw_and_ranked_predictions(tmp_path):
    import pickle

    con = init_ml_db(tmp_path / "ml.duckdb")
    cache_dir = tmp_path / "cache" / "run_id=run" / "fold_id=wf"
    cache_dir.mkdir(parents=True)
    sparse.save_npz(cache_dir / "X_test.npz", sparse.csr_matrix(np.array([[1.0], [2.0]], dtype=np.float32)))
    con.register(
        "metadata",
        __import__("pandas").DataFrame(
            {
                "trade_date": ["2024-01-02", "2024-01-02"],
                "code": ["000001.SZ", "000002.SZ"],
                "is_bse": [False, False],
            }
        ),
    )
    con.execute(f"copy metadata to '{cache_dir / 'metadata_test.parquet'}' (format parquet)")
    con.unregister("metadata")
    schema = FeatureSchema("vpa_d_sequence", ["ret_1d"], [], ["ret_1d"], {}, {"ret_1d": 0.0}, "v2")
    save_feature_schema(schema, cache_dir / "feature_schema.json")
    (cache_dir / "manifest.json").write_text(json.dumps({"test_rows": 2}), encoding="utf-8")

    artifacts = []
    for name, model in [
        ("abs", LinearFallbackModel(["ret_1d"], {"ret_1d": 1.0})),
        ("active", LinearFallbackModel(["ret_1d"], {"ret_1d": 2.0})),
        ("risk", LogisticFallbackModel(["ret_1d"], {"ret_1d": 1.0})),
    ]:
        path = tmp_path / f"{name}.pkl"
        with path.open("wb") as handle:
            pickle.dump(model, handle)
        artifacts.append(SimpleArtifact(f"{name}_model", path, cache_dir / "feature_schema.json"))

    cache = FoldMatrixCache.from_paths("run", "wf", tmp_path / "cache")
    rows = write_chunked_fold_predictions(
        con,
        cache,
        artifacts[0],
        artifacts[1],
        artifacts[2],
        score_version="v2_three_model",
        chunk_size=1,
    )

    raw_count = con.execute("select count(*) from ml_prediction_raw_daily").fetchone()[0]
    ranked = con.execute("select count(*), min(trade_score_v2), max(trade_score_v2) from ml_predictions_daily").fetchone()
    assert rows == 2
    assert raw_count == 2
    assert ranked[0] == 2
    assert ranked[1] is not None
    assert ranked[2] is not None


class SimpleArtifact:
    def __init__(self, model_id, artifact_uri, feature_schema_uri):
        self.model_id = model_id
        self.feature_set_id = "vpa_d_sequence"
        self.horizon_d = 5
        self.artifact_uri = artifact_uri
        self.feature_schema_uri = feature_schema_uri

