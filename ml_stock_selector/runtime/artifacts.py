from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
from typing import Any

import duckdb
import pandas as pd


def prepare_run_artifact_dir(
    output_root: Path | str,
    run_id: str,
    *,
    config_path: Path | str | None = None,
    run_manifest: dict[str, Any] | None = None,
) -> Path:
    root = Path(output_root) / str(run_id)
    root.mkdir(parents=True, exist_ok=True)
    if config_path is not None:
        source = Path(config_path)
        if source.exists():
            shutil.copy2(source, root / "config_snapshot.toml")
            (root / "config_hash.txt").write_text(_file_sha256(source) + "\n", encoding="utf-8")
    manifest = {
        "run_id": run_id,
        "created_at": _now(),
        "git_commit": _git_commit(),
        **(run_manifest or {}),
    }
    _write_json(root / "run_manifest.json", manifest)
    return root


def write_backtest_fold_artifacts(
    run_root: Path | str,
    *,
    fold_id: str,
    strategy_id: str,
    score_version: str,
    portfolio_id: str,
    backtest_params: dict[str, Any],
    targets: pd.DataFrame,
    diagnostics: pd.DataFrame,
    orders: pd.DataFrame,
    positions: pd.DataFrame,
    nav: pd.DataFrame,
    metrics: pd.DataFrame,
) -> Path:
    fold_root = Path(run_root) / "folds" / fold_id
    backtest_root = fold_root / "backtest" / f"strategy_id={strategy_id}" / f"score_version={score_version}"
    portfolio_root = fold_root / "portfolio" / f"portfolio_id={portfolio_id}" / f"score_version={score_version}"
    backtest_root.mkdir(parents=True, exist_ok=True)
    portfolio_root.mkdir(parents=True, exist_ok=True)
    _write_json(
        fold_root / "fold_manifest.json",
        {
            "fold_id": fold_id,
            "updated_at": _now(),
            "strategy_id": strategy_id,
            "score_version": score_version,
            "portfolio_id": portfolio_id,
        },
    )
    _write_json(backtest_root / "backtest_params.json", _jsonable(backtest_params))
    _write_parquet(targets, portfolio_root / "targets.parquet")
    _write_parquet(diagnostics, portfolio_root / "diagnostics.parquet")
    _write_parquet(orders, backtest_root / "orders.parquet")
    _write_parquet(positions, backtest_root / "positions.parquet")
    _write_parquet(nav, backtest_root / "nav.parquet")
    _write_json(backtest_root / "metrics.json", _records(metrics))
    return backtest_root


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    con = duckdb.connect(":memory:")
    try:
        con.register("_artifact_frame", frame)
        con.execute("copy _artifact_frame to ? (format parquet)", [str(path)])
    finally:
        con.close()


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    return [_jsonable(row) for row in frame.to_dict("records")]


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if pd.isna(value) if value is not None and not isinstance(value, (dict, list, tuple)) else False:
        return None
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
