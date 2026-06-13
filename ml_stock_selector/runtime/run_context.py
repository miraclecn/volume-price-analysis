from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
from typing import Any

import pandas as pd

from ml_stock_selector.storage import upsert_dataframe


@dataclass(frozen=True)
class RunContext:
    run_id: str
    run_type: str
    experiment_name: str | None
    config_path: Path | None
    config_hash: str | None
    git_commit: str | None
    artifact_root: Path
    feature_set_id: str | None
    label_version: str | None
    score_version: str | None
    alpha_data_db: str | None = None
    alpha_data_latest_date: str | None = None
    vpa_db: str | None = None
    ml_db: str | None = None
    feature_store_version: str | None = None
    created_at: str = ""
    started_at: str | None = None
    finished_at: str | None = None
    status: str = "created"
    notes: str | None = None


def create_run_context(
    *,
    run_type: str,
    run_id: str,
    artifact_root: Path | str,
    experiment_name: str | None = None,
    config_path: Path | str | None = None,
    alpha_data_db: str | None = None,
    alpha_data_latest_date: str | None = None,
    vpa_db: str | None = None,
    ml_db: str | None = None,
    feature_set_id: str | None = None,
    feature_store_version: str | None = None,
    label_version: str | None = None,
    score_version: str | None = None,
    notes: str | None = None,
    status: str = "created",
) -> RunContext:
    config = Path(config_path) if config_path is not None else None
    root = Path(artifact_root) / run_id
    root.mkdir(parents=True, exist_ok=True)
    config_hash = _file_sha256(config) if config is not None and config.exists() else None
    git_commit = _git_commit()
    created_at = _now()
    context = RunContext(
        run_id=run_id,
        run_type=run_type,
        experiment_name=experiment_name,
        config_path=config,
        config_hash=config_hash,
        git_commit=git_commit,
        artifact_root=root,
        feature_set_id=feature_set_id,
        label_version=label_version,
        score_version=score_version,
        alpha_data_db=alpha_data_db,
        alpha_data_latest_date=alpha_data_latest_date,
        vpa_db=vpa_db,
        ml_db=ml_db,
        feature_store_version=feature_store_version,
        created_at=created_at,
        started_at=created_at,
        status=status,
        notes=notes,
    )
    _write_run_files(context)
    return context


def register_run_context(con, context: RunContext) -> None:
    upsert_dataframe(con, "ml_runs", pd.DataFrame([_run_row(context)]), ["run_id"])


def register_run_fold(con, context: RunContext, fold: dict[str, Any], *, status: str = "created") -> Path:
    fold_id = str(fold["fold_id"])
    fold_dir = context.artifact_root / "folds" / fold_id
    fold_dir.mkdir(parents=True, exist_ok=True)
    now = _now()
    payload = {
        "run_id": context.run_id,
        "fold_id": fold_id,
        "train_start": _string_or_none(fold.get("train_start")),
        "train_end": _string_or_none(fold.get("train_end")),
        "valid_start": _string_or_none(fold.get("valid_start")),
        "valid_end": _string_or_none(fold.get("valid_end")),
        "test_start": _string_or_none(fold.get("test_start")),
        "test_end": _string_or_none(fold.get("test_end")),
        "gap_type": _string_or_none(fold.get("gap_type")),
        "embargo_days": int(fold["embargo_days"]) if fold.get("embargo_days") is not None else None,
        "status": status,
        "artifact_dir": str(fold_dir),
        "created_at": now,
    }
    manifest_path = fold_dir / "fold_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    manifest.update(payload)
    _write_json(manifest_path, manifest)
    upsert_dataframe(con, "ml_run_folds", pd.DataFrame([payload]), ["run_id", "fold_id"])
    return fold_dir


def update_run_status(con, context: RunContext, status: str) -> None:
    con.execute(
        "update ml_runs set status = ?, finished_at = ? where run_id = ?",
        [status, _now() if status in {"success", "failed"} else None, context.run_id],
    )


def _write_run_files(context: RunContext) -> None:
    if context.config_path is not None and context.config_path.exists():
        shutil.copy2(context.config_path, context.artifact_root / "config_snapshot.toml")
    if context.config_hash is not None:
        (context.artifact_root / "config_hash.txt").write_text(context.config_hash + "\n", encoding="utf-8")
    if context.git_commit is not None:
        (context.artifact_root / "git_commit.txt").write_text(context.git_commit + "\n", encoding="utf-8")
    _write_json(context.artifact_root / "run_manifest.json", _run_row(context))


def _run_row(context: RunContext) -> dict[str, Any]:
    payload = asdict(context)
    payload["config_path"] = str(context.config_path) if context.config_path is not None else None
    payload["artifact_root"] = str(context.artifact_root)
    return payload


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


def _string_or_none(value: object) -> str | None:
    return None if value is None else str(value)
