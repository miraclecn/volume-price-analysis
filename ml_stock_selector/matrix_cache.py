from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import json
import shutil
from uuid import uuid4

import numpy as np
import pandas as pd
from scipy import sparse

from ml_stock_selector.feature_matrix import FeatureSchema as MatrixFeatureSchema
from ml_stock_selector.feature_matrix import save_feature_schema
from ml_stock_selector.feature_store_reader import FeatureStoreSpec, iter_feature_store_batches, load_feature_schema
from ml_stock_selector.universe import apply_universe_filter

FOLD_MANIFEST_STAGES = [
    "pending",
    "matrix_built",
    "models_trained",
    "predicted",
    "backtested",
    "metrics_written",
    "failed",
]


@dataclass(frozen=True)
class FoldMatrixCache:
    run_id: str
    fold_id: str
    cache_dir: Path
    x_train_path: Path
    x_valid_path: Path
    x_test_path: Path
    x_train_dense_path: Path
    x_valid_dense_path: Path
    x_test_dense_path: Path
    y_abs_train_path: Path
    y_abs_valid_path: Path
    y_active_train_path: Path
    y_active_valid_path: Path
    y_risk_train_path: Path
    y_risk_valid_path: Path
    group_train_path: Path
    group_valid_path: Path
    metadata_train_path: Path
    metadata_valid_path: Path
    metadata_test_path: Path
    feature_schema_path: Path
    manifest_path: Path

    @classmethod
    def from_paths(cls, run_id: str, fold_id: str, cache_root: Path | str) -> "FoldMatrixCache":
        cache_dir = Path(cache_root) / f"run_id={run_id}" / f"fold_id={fold_id}"
        return cls(
            run_id=run_id,
            fold_id=fold_id,
            cache_dir=cache_dir,
            x_train_path=cache_dir / "X_train.npz",
            x_valid_path=cache_dir / "X_valid.npz",
            x_test_path=cache_dir / "X_test.npz",
            x_train_dense_path=cache_dir / "X_train.npy",
            x_valid_dense_path=cache_dir / "X_valid.npy",
            x_test_dense_path=cache_dir / "X_test.npy",
            y_abs_train_path=cache_dir / "y_abs_train.npy",
            y_abs_valid_path=cache_dir / "y_abs_valid.npy",
            y_active_train_path=cache_dir / "y_active_train.npy",
            y_active_valid_path=cache_dir / "y_active_valid.npy",
            y_risk_train_path=cache_dir / "y_risk_train.npy",
            y_risk_valid_path=cache_dir / "y_risk_valid.npy",
            group_train_path=cache_dir / "group_train.npy",
            group_valid_path=cache_dir / "group_valid.npy",
            metadata_train_path=cache_dir / "metadata_train.parquet",
            metadata_valid_path=cache_dir / "metadata_valid.parquet",
            metadata_test_path=cache_dir / "metadata_test.parquet",
            feature_schema_path=cache_dir / "feature_schema.json",
            manifest_path=cache_dir / "manifest.json",
        )


def build_fold_matrix_cache(
    con,
    feature_store_spec: FeatureStoreSpec,
    fold_config,
    run_id: str,
    feature_set_id: str,
    horizon_d: int,
    label_base: str,
    universe_config,
    cache_root: str | Path,
    batch_size: int = 50000,
) -> FoldMatrixCache:
    fold_id = str(_fold_value(fold_config, "fold_id"))
    cache = FoldMatrixCache.from_paths(run_id, fold_id, cache_root)
    cache.cache_dir.mkdir(parents=True, exist_ok=True)
    schema = load_feature_schema(feature_store_spec)
    universe = dict(universe_config)
    exclude_bse = bool(universe.get("exclude_bse", False))
    min_adv20_amount = universe.get("min_adv20_amount")

    train_rows = _write_labeled_split_streaming(
        con,
        cache.x_train_dense_path,
        cache.metadata_train_path,
        {
            "absolute": cache.y_abs_train_path,
            "active": cache.y_active_train_path,
            "risk": cache.y_risk_train_path,
            "group": cache.group_train_path,
        },
        feature_store_spec,
        str(_fold_value(fold_config, "train_start")),
        str(_fold_value(fold_config, "train_end")),
        horizon_d,
        label_base,
        exclude_bse,
        min_adv20_amount,
        batch_size,
        schema.numeric_columns,
    )
    valid_rows = _write_labeled_split_streaming(
        con,
        cache.x_valid_dense_path,
        cache.metadata_valid_path,
        {
            "absolute": cache.y_abs_valid_path,
            "active": cache.y_active_valid_path,
            "risk": cache.y_risk_valid_path,
            "group": cache.group_valid_path,
        },
        feature_store_spec,
        str(_fold_value(fold_config, "valid_start")),
        str(_fold_value(fold_config, "valid_end")),
        horizon_d,
        label_base,
        exclude_bse,
        min_adv20_amount,
        batch_size,
        schema.numeric_columns,
    )
    test_rows = _write_unlabeled_split_streaming(
        con,
        cache.x_test_dense_path,
        cache.metadata_test_path,
        feature_store_spec,
        str(_fold_value(fold_config, "test_start")),
        str(_fold_value(fold_config, "test_end")),
        exclude_bse,
        batch_size,
        schema.numeric_columns,
    )
    save_feature_schema(
        MatrixFeatureSchema(
            feature_set_id=feature_set_id,
            numeric_columns=schema.numeric_columns,
            categorical_columns=[],
            output_columns=schema.numeric_columns,
            category_levels={},
            fill_values={col: 0.0 for col in schema.numeric_columns},
            schema_version=schema.schema_version,
        ),
        cache.feature_schema_path,
    )
    _write_manifest(
        cache.manifest_path,
        {
            "run_id": run_id,
            "fold_id": fold_id,
            "status": "matrix_built",
            "feature_store_version": feature_store_spec.dataset_version,
            "feature_schema_hash": schema.schema_hash,
            "feature_set_id": feature_set_id,
            "horizon_d": horizon_d,
            "label_base": label_base,
            "train_rows": train_rows,
            "valid_rows": valid_rows,
            "test_rows": test_rows,
            "exclude_bse": exclude_bse,
            "train_filter_can_buy_next_open": True,
            "train_filter_min_adv20_amount": min_adv20_amount,
            "matrix_format": "dense_float32_npy",
            "train_start": str(_fold_value(fold_config, "train_start")),
            "train_end": str(_fold_value(fold_config, "train_end")),
            "valid_start": str(_fold_value(fold_config, "valid_start")),
            "valid_end": str(_fold_value(fold_config, "valid_end")),
            "test_start": str(_fold_value(fold_config, "test_start")),
            "test_end": str(_fold_value(fold_config, "test_end")),
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    return cache


def read_fold_manifest(path: Path | str) -> dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def update_fold_manifest_status(path: Path | str, status: str, **updates: object) -> dict[str, object]:
    path = Path(path)
    manifest = read_fold_manifest(path) if path.exists() else {}
    manifest.update(updates)
    manifest["status"] = status
    if status == "failed":
        manifest.setdefault("failed_at", datetime.now(timezone.utc).isoformat())
    else:
        manifest[f"{status}_at"] = datetime.now(timezone.utc).isoformat()
    _write_manifest(path, manifest)
    return manifest


def mark_fold_manifest_failed(path: Path | str, error: BaseException) -> dict[str, object]:
    return update_fold_manifest_status(
        path,
        "failed",
        failed_at=datetime.now(timezone.utc).isoformat(),
        error_message=str(error),
    )


def is_fold_manifest_complete(cache: FoldMatrixCache, stage: str) -> bool:
    if not cache.manifest_path.exists():
        return False
    manifest = read_fold_manifest(cache.manifest_path)
    status = str(manifest.get("status", "pending"))
    if status == "failed":
        return False
    if _stage_index(status) < _stage_index(stage):
        return False
    return _required_paths_exist(cache, stage)


def _stage_index(stage: str) -> int:
    if stage not in FOLD_MANIFEST_STAGES:
        raise ValueError(f"Unknown fold manifest stage: {stage}")
    return FOLD_MANIFEST_STAGES.index(stage)


def _required_paths_for_stage(cache: FoldMatrixCache, stage: str) -> list[Path]:
    matrix_paths = [
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
    ]
    if _stage_index(stage) <= _stage_index("matrix_built"):
        return matrix_paths
    return matrix_paths + [cache.manifest_path]


def _required_paths_exist(cache: FoldMatrixCache, stage: str) -> bool:
    matrix_exists = (
        _matrix_path_exists(cache.x_train_dense_path, cache.x_train_path)
        and _matrix_path_exists(cache.x_valid_dense_path, cache.x_valid_path)
        and _matrix_path_exists(cache.x_test_dense_path, cache.x_test_path)
    )
    other_paths = [
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
    ]
    if _stage_index(stage) > _stage_index("matrix_built"):
        other_paths.append(cache.manifest_path)
    return matrix_exists and all(path.exists() for path in other_paths)


def _matrix_path_exists(dense_path: Path, sparse_path: Path) -> bool:
    return dense_path.exists() or sparse_path.exists()


def load_cached_matrix(dense_path: Path, sparse_path: Path):
    if dense_path.exists():
        return np.load(dense_path, mmap_mode="r")
    return sparse.load_npz(sparse_path)


def load_train_matrix(cache: FoldMatrixCache):
    return load_cached_matrix(cache.x_train_dense_path, cache.x_train_path)


def load_valid_matrix(cache: FoldMatrixCache):
    return load_cached_matrix(cache.x_valid_dense_path, cache.x_valid_path)


def load_test_matrix(cache: FoldMatrixCache):
    return load_cached_matrix(cache.x_test_dense_path, cache.x_test_path)


def _build_labeled_split(
    con,
    spec: FeatureStoreSpec,
    start_date: str,
    end_date: str,
    horizon_d: int,
    label_base: str,
    exclude_bse: bool,
    min_adv20_amount: object | None,
    batch_size: int,
) -> pd.DataFrame:
    labels = _labels_for_range(con, start_date, end_date, horizon_d, label_base)
    metadata = _metadata_for_range(con, start_date, end_date)
    frames = []
    for features in iter_feature_store_batches(spec, start_date, end_date, batch_size=batch_size):
        merged = features.merge(labels, on=["trade_date", "code"], how="inner").merge(metadata, on=["trade_date", "code"], how="left")
        merged = apply_universe_filter(merged, exclude_bse=exclude_bse)
        merged = _filter_trainable_tradeability(merged, min_adv20_amount)
        merged = merged.dropna(subset=["absolute_label", "active_label", "risk_label"])
        frames.append(merged)
    if not frames:
        return pd.DataFrame(
            columns=[
                "trade_date",
                "code",
                "absolute_label",
                "active_label",
                "risk_label",
                "industry_code",
                "industry_name",
                "is_bse",
                "is_st",
                "is_paused",
                "adv20_amount",
                "can_buy_next_open",
            ]
        )
    return _concat_sorted(frames)


def _write_labeled_split_streaming(
    con,
    x_path: Path,
    metadata_path: Path,
    label_paths: dict[str, Path],
    spec: FeatureStoreSpec,
    start_date: str,
    end_date: str,
    horizon_d: int,
    label_base: str,
    exclude_bse: bool,
    min_adv20_amount: object | None,
    batch_size: int,
    feature_columns: list[str],
) -> int:
    labels = _labels_for_range(con, start_date, end_date, horizon_d, label_base)
    metadata = _metadata_for_range(con, start_date, end_date)
    temp_dir = x_path.parent / f".{x_path.stem}_{uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    x_chunks: list[Path] = []
    abs_chunks: list[Path] = []
    active_chunks: list[Path] = []
    risk_chunks: list[Path] = []
    metadata_chunks: list[Path] = []
    groups = _GroupAccumulator()
    row_count = 0
    try:
        for idx, features in enumerate(iter_feature_store_batches(spec, start_date, end_date, batch_size=batch_size)):
            merged = (
                features.merge(labels, on=["trade_date", "code"], how="inner")
                .merge(metadata, on=["trade_date", "code"], how="left")
            )
            merged = apply_universe_filter(merged, exclude_bse=exclude_bse)
            merged = _filter_trainable_tradeability(merged, min_adv20_amount)
            merged = merged.dropna(subset=["absolute_label", "active_label", "risk_label"])
            if merged.empty:
                continue
            merged = merged.sort_values(["trade_date", "code"]).reset_index(drop=True)
            chunk_suffix = f"{idx:06d}"
            x_chunk = temp_dir / f"X_{chunk_suffix}.npy"
            np.save(x_chunk, _frame_to_dense_matrix(merged, feature_columns))
            x_chunks.append(x_chunk)
            abs_chunk = temp_dir / f"y_abs_{chunk_suffix}.npy"
            active_chunk = temp_dir / f"y_active_{chunk_suffix}.npy"
            risk_chunk = temp_dir / f"y_risk_{chunk_suffix}.npy"
            np.save(abs_chunk, merged["absolute_label"].to_numpy(dtype=np.float32))
            np.save(active_chunk, merged["active_label"].to_numpy(dtype=np.float32))
            np.save(risk_chunk, merged["risk_label"].to_numpy(dtype=np.float32))
            abs_chunks.append(abs_chunk)
            active_chunks.append(active_chunk)
            risk_chunks.append(risk_chunk)
            metadata_chunk = temp_dir / f"metadata_{chunk_suffix}.parquet"
            _copy_dataframe_to_parquet(con, _metadata_frame(merged), metadata_chunk)
            metadata_chunks.append(metadata_chunk)
            groups.add(merged["trade_date"])
            row_count += len(merged)
        _finalize_dense_chunks(x_chunks, x_path, feature_columns)
        _finalize_label_chunks(abs_chunks, label_paths["absolute"])
        _finalize_label_chunks(active_chunks, label_paths["active"])
        _finalize_label_chunks(risk_chunks, label_paths["risk"])
        np.save(label_paths["group"], groups.to_array())
        _finalize_metadata_chunks(con, metadata_chunks, metadata_path)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
    return row_count


def _build_unlabeled_split(
    con,
    spec: FeatureStoreSpec,
    start_date: str,
    end_date: str,
    exclude_bse: bool,
    batch_size: int,
) -> pd.DataFrame:
    metadata = _metadata_for_range(con, start_date, end_date)
    frames = []
    for features in iter_feature_store_batches(spec, start_date, end_date, batch_size=batch_size):
        merged = features.merge(metadata, on=["trade_date", "code"], how="left")
        merged = apply_universe_filter(merged, exclude_bse=exclude_bse)
        frames.append(merged)
    return _concat_sorted(frames)


def _write_unlabeled_split_streaming(
    con,
    x_path: Path,
    metadata_path: Path,
    spec: FeatureStoreSpec,
    start_date: str,
    end_date: str,
    exclude_bse: bool,
    batch_size: int,
    feature_columns: list[str],
) -> int:
    metadata = _metadata_for_range(con, start_date, end_date)
    temp_dir = x_path.parent / f".{x_path.stem}_{uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    x_chunks: list[Path] = []
    metadata_chunks: list[Path] = []
    row_count = 0
    try:
        for idx, features in enumerate(iter_feature_store_batches(spec, start_date, end_date, batch_size=batch_size)):
            merged = features.merge(metadata, on=["trade_date", "code"], how="left")
            merged = apply_universe_filter(merged, exclude_bse=exclude_bse)
            if merged.empty:
                continue
            merged = merged.sort_values(["trade_date", "code"]).reset_index(drop=True)
            chunk_suffix = f"{idx:06d}"
            x_chunk = temp_dir / f"X_{chunk_suffix}.npy"
            np.save(x_chunk, _frame_to_dense_matrix(merged, feature_columns))
            x_chunks.append(x_chunk)
            metadata_chunk = temp_dir / f"metadata_{chunk_suffix}.parquet"
            _copy_dataframe_to_parquet(con, _metadata_frame(merged), metadata_chunk)
            metadata_chunks.append(metadata_chunk)
            row_count += len(merged)
        _finalize_dense_chunks(x_chunks, x_path, feature_columns)
        _finalize_metadata_chunks(con, metadata_chunks, metadata_path)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
    return row_count


def _labels_for_range(con, start_date: str, end_date: str, horizon_d: int, label_base: str) -> pd.DataFrame:
    return con.execute(
        """
        select trade_date, code, absolute_label, active_label, risk_label
        from ml_labels_daily
        where trade_date >= ? and trade_date <= ?
          and horizon_d = ? and label_base = ?
          and absolute_label is not null
          and active_label is not null
          and risk_label is not null
        """,
        [start_date, end_date, horizon_d, label_base],
    ).fetchdf()


def _metadata_for_range(con, start_date: str, end_date: str) -> pd.DataFrame:
    return con.execute(
        """
        select trade_date, code, industry_code, industry_name, is_bse, is_st, is_paused,
               adv20_amount, can_buy_next_open
        from ml_tradeability_daily
        where trade_date >= ? and trade_date <= ?
        """,
        [start_date, end_date],
    ).fetchdf()


def _filter_trainable_tradeability(frame: pd.DataFrame, min_adv20_amount: object | None) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    out = frame.copy()
    mask = pd.Series(True, index=out.index)
    if "can_buy_next_open" in out:
        mask &= out["can_buy_next_open"].fillna(False).astype(bool)
    for column in ["is_st", "is_paused"]:
        if column in out:
            mask &= ~out[column].fillna(False).astype(bool)
    if min_adv20_amount is not None and "adv20_amount" in out:
        threshold = float(min_adv20_amount)
        if threshold > 0.0:
            mask &= pd.to_numeric(out["adv20_amount"], errors="coerce").fillna(0.0) >= threshold
    return out[mask].copy()


def _frame_to_sparse_matrix(frame: pd.DataFrame, feature_columns: list[str]):
    if frame.empty:
        return sparse.csr_matrix((0, len(feature_columns)), dtype=np.float32)
    return sparse.csr_matrix(frame[feature_columns].fillna(0.0).to_numpy(dtype=np.float32))


def _frame_to_dense_matrix(frame: pd.DataFrame, feature_columns: list[str]) -> np.ndarray:
    if frame.empty:
        return np.empty((0, len(feature_columns)), dtype=np.float32)
    return frame[feature_columns].fillna(0.0).to_numpy(dtype=np.float32)


def _save_split(path: Path, metadata_path: Path, con, frame: pd.DataFrame, feature_columns: list[str]) -> None:
    if frame.empty:
        matrix = sparse.csr_matrix((0, len(feature_columns)), dtype=np.float32)
    else:
        matrix = _frame_to_sparse_matrix(frame, feature_columns)
    sparse.save_npz(path, matrix)
    metadata = _metadata_frame(frame)
    _copy_dataframe_to_parquet(con, metadata, metadata_path)


def _metadata_frame(frame: pd.DataFrame) -> pd.DataFrame:
    metadata_columns = [
        "trade_date",
        "code",
        "industry_code",
        "industry_name",
        "is_bse",
        "is_st",
        "is_paused",
        "adv20_amount",
        "can_buy_next_open",
    ]
    if frame.empty:
        return pd.DataFrame(columns=metadata_columns)
    return frame[[col for col in metadata_columns if col in frame.columns]].copy()


def _finalize_sparse_chunks(chunks: list[Path], output_path: Path, feature_columns: list[str]) -> None:
    if not chunks:
        sparse.save_npz(output_path, sparse.csr_matrix((0, len(feature_columns)), dtype=np.float32))
        return
    matrices = [sparse.load_npz(path) for path in chunks]
    sparse.save_npz(output_path, sparse.vstack(matrices, format="csr"))


def _finalize_dense_chunks(chunks: list[Path], output_path: Path, feature_columns: list[str]) -> None:
    if not chunks:
        np.save(output_path, np.empty((0, len(feature_columns)), dtype=np.float32))
        return
    shapes = [np.load(path, mmap_mode="r").shape for path in chunks]
    total_rows = int(sum(shape[0] for shape in shapes))
    col_count = len(feature_columns)
    out = np.lib.format.open_memmap(output_path, mode="w+", dtype=np.float32, shape=(total_rows, col_count))
    offset = 0
    for path, shape in zip(chunks, shapes):
        rows = int(shape[0])
        if rows:
            out[offset : offset + rows] = np.load(path, mmap_mode="r")
        offset += rows
    del out


def _finalize_label_chunks(chunks: list[Path], output_path: Path) -> None:
    if not chunks:
        np.save(output_path, np.array([], dtype=np.float32))
        return
    np.save(output_path, np.concatenate([np.load(path) for path in chunks]).astype(np.float32, copy=False))


def _finalize_metadata_chunks(con, chunks: list[Path], output_path: Path) -> None:
    if not chunks:
        _copy_dataframe_to_parquet(con, _metadata_frame(pd.DataFrame()), output_path)
        return
    glob = str(chunks[0].parent / "metadata_*.parquet")
    temp_name = f"_matrix_metadata_{uuid4().hex}"
    con.execute(
        f"""
        create temporary table {temp_name} as
        select *
        from read_parquet(?)
        order by trade_date, code
        """,
        [glob],
    )
    try:
        con.execute(f"copy {temp_name} to ? (format parquet, compression zstd)", [str(output_path)])
    finally:
        con.execute(f"drop table if exists {temp_name}")


def _copy_dataframe_to_parquet(con, frame: pd.DataFrame, path: Path) -> None:
    temp_name = f"_matrix_cache_{uuid4().hex}"
    con.register(temp_name, frame)
    try:
        con.execute(f"copy {temp_name} to ? (format parquet, compression zstd)", [str(path)])
    finally:
        con.unregister(temp_name)


def _groups_for_dates(dates: pd.Series) -> np.ndarray:
    if dates.empty:
        return np.array([], dtype=np.int32)
    return dates.astype(str).groupby(dates.astype(str), sort=True).size().to_numpy(dtype=np.int32)


class _GroupAccumulator:
    def __init__(self) -> None:
        self._groups: list[int] = []
        self._last_date: str | None = None

    def add(self, dates: pd.Series) -> None:
        if dates.empty:
            return
        counts = dates.astype(str).groupby(dates.astype(str), sort=True).size()
        for date, count in counts.items():
            value = int(count)
            if self._last_date == str(date) and self._groups:
                self._groups[-1] += value
            else:
                self._groups.append(value)
                self._last_date = str(date)

    def to_array(self) -> np.ndarray:
        return np.array(self._groups, dtype=np.int32)


def _concat_sorted(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values(["trade_date", "code"]).reset_index(drop=True)


def _fold_value(fold_config, key: str):
    if isinstance(fold_config, dict):
        return fold_config[key]
    return getattr(fold_config, key)


def _write_manifest(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
