from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json


def register_model(
    con,
    *,
    model_id: str,
    model_type: str,
    feature_set_id: str,
    label_name: str,
    label_base: str,
    horizon_d: int,
    artifact_uri: str,
    feature_schema_uri: str,
    params_json: str = "{}",
    metrics_json: str = "{}",
    notes: str | None = None,
    run_id: str | None = None,
    fold_id: str | None = None,
    feature_store_version: str | None = None,
    feature_schema_hash: str | None = None,
    train_start: str | None = None,
    train_end: str | None = None,
    valid_start: str | None = None,
    valid_end: str | None = None,
    test_start: str | None = None,
    test_end: str | None = None,
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    _ensure_registry_columns(con)
    con.execute(
        """
        insert into ml_model_registry (
            model_id, model_type, feature_set_id, label_name, label_base, horizon_d,
            run_id, fold_id, feature_store_version, feature_schema_hash,
            train_start, train_end, valid_start, valid_end, test_start, test_end,
            params_json, metrics_json, feature_schema_uri, artifact_uri,
            is_active, activated_at, deactivated_at, created_at, notes
        ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, false, null, null, ?, ?)
        on conflict (model_id) do update set
            run_id = excluded.run_id,
            fold_id = excluded.fold_id,
            feature_store_version = excluded.feature_store_version,
            feature_schema_hash = excluded.feature_schema_hash,
            train_start = excluded.train_start,
            train_end = excluded.train_end,
            valid_start = excluded.valid_start,
            valid_end = excluded.valid_end,
            test_start = excluded.test_start,
            test_end = excluded.test_end,
            params_json = excluded.params_json,
            metrics_json = excluded.metrics_json,
            feature_schema_uri = excluded.feature_schema_uri,
            artifact_uri = excluded.artifact_uri,
            notes = excluded.notes
        """,
        [
            model_id,
            model_type,
            feature_set_id,
                label_name,
                label_base,
                horizon_d,
                run_id,
                fold_id,
                feature_store_version,
                feature_schema_hash,
                train_start,
                train_end,
                valid_start,
                valid_end,
                test_start,
                test_end,
                params_json,
                metrics_json,
                feature_schema_uri,
            artifact_uri,
            now,
            notes,
        ],
    )


def _ensure_registry_columns(con) -> None:
    for sql in [
        "alter table ml_model_registry add column if not exists run_id varchar",
        "alter table ml_model_registry add column if not exists fold_id varchar",
        "alter table ml_model_registry add column if not exists feature_store_version varchar",
        "alter table ml_model_registry add column if not exists feature_schema_hash varchar",
    ]:
        con.execute(sql)


def register_model_bundle(
    con,
    *,
    bundle_id: str,
    run_id: str,
    bundle_role: str,
    absolute_model_id: str,
    active_model_id: str,
    risk_model_id: str,
    feature_set_id: str,
    label_base: str,
    horizon_d: int,
    score_version: str,
    artifact_dir: str,
    fold_id: str | None = None,
    status: str = "candidate",
    notes: str | None = None,
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    con.execute(
        """
        insert into ml_model_bundles (
            bundle_id, run_id, fold_id, bundle_role,
            absolute_model_id, active_model_id, risk_model_id,
            feature_set_id, label_base, horizon_d, score_version,
            artifact_dir, status, created_at, activated_at, deactivated_at, notes
        ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, null, null, ?)
        on conflict (bundle_id) do update set
            run_id = excluded.run_id,
            fold_id = excluded.fold_id,
            bundle_role = excluded.bundle_role,
            absolute_model_id = excluded.absolute_model_id,
            active_model_id = excluded.active_model_id,
            risk_model_id = excluded.risk_model_id,
            feature_set_id = excluded.feature_set_id,
            label_base = excluded.label_base,
            horizon_d = excluded.horizon_d,
            score_version = excluded.score_version,
            artifact_dir = excluded.artifact_dir,
            status = excluded.status,
            notes = excluded.notes
        """,
        [
            bundle_id,
            run_id,
            fold_id,
            bundle_role,
            absolute_model_id,
            active_model_id,
            risk_model_id,
            feature_set_id,
            label_base,
            horizon_d,
            score_version,
            artifact_dir,
            status,
            now,
            notes,
        ],
    )


def get_model_bundle(con, bundle_id: str) -> dict[str, object]:
    frame = con.execute("select * from ml_model_bundles where bundle_id = ?", [bundle_id]).fetchdf()
    if frame.empty:
        raise ValueError(f"Unknown bundle_id: {bundle_id}")
    return frame.iloc[0].to_dict()


def get_active_model_bundle(
    con,
    bundle_role: str,
    feature_set_id: str,
    label_base: str,
    horizon_d: int,
) -> dict[str, object]:
    frame = con.execute(
        """
        select *
        from ml_model_bundles
        where bundle_role = ?
          and feature_set_id = ?
          and label_base = ?
          and horizon_d = ?
          and status = 'active'
        order by activated_at desc nulls last, created_at desc nulls last
        """,
        [bundle_role, feature_set_id, label_base, horizon_d],
    ).fetchdf()
    if frame.empty:
        raise ValueError("No active model bundle found")
    return frame.iloc[0].to_dict()


def activate_model_bundle(con, bundle_id: str) -> None:
    bundle = get_model_bundle(con, bundle_id)
    _validate_bundle_models(con, bundle)
    now = datetime.now(timezone.utc).isoformat()
    con.execute(
        """
        update ml_model_bundles
        set status = 'retired', deactivated_at = ?
        where bundle_role = ?
          and feature_set_id = ?
          and label_base = ?
          and horizon_d = ?
          and status = 'active'
          and bundle_id <> ?
        """,
        [
            now,
            bundle["bundle_role"],
            bundle["feature_set_id"],
            bundle["label_base"],
            int(bundle["horizon_d"]),
            bundle_id,
        ],
    )
    con.execute(
        """
        update ml_model_bundles
        set status = 'active', activated_at = ?, deactivated_at = null
        where bundle_id = ?
        """,
        [now, bundle_id],
    )
    activate_model(con, str(bundle["absolute_model_id"]))
    activate_model(con, str(bundle["active_model_id"]))
    activate_model(con, str(bundle["risk_model_id"]))


def _validate_bundle_models(con, bundle: dict[str, object]) -> None:
    expected = {
        str(bundle["absolute_model_id"]): ("alpha_ranker", "absolute_label"),
        str(bundle["active_model_id"]): ("active_ranker", "active_label"),
        str(bundle["risk_model_id"]): ("risk_model", "risk_label"),
    }
    placeholders = ",".join("?" for _ in expected)
    rows = con.execute(
        f"""
        select model_id, model_type, feature_set_id, label_name, label_base, horizon_d, feature_schema_uri
        from ml_model_registry
        where model_id in ({placeholders})
        """,
        list(expected),
    ).fetchall()
    found = {str(row[0]): row for row in rows}
    missing = [model_id for model_id in expected if model_id not in found]
    if missing:
        raise ValueError(f"Bundle component model missing: {', '.join(missing)}")
    schema_hashes = set()
    for model_id, (model_type, label_name) in expected.items():
        row = found[model_id]
        if str(row[1]) != model_type or str(row[3]) != label_name:
            raise ValueError(f"Bundle component model has wrong role: {model_id}")
        if str(row[2]) != str(bundle["feature_set_id"]) or str(row[4]) != str(bundle["label_base"]) or int(row[5]) != int(bundle["horizon_d"]):
            raise ValueError(f"Bundle component model metadata mismatch: {model_id}")
        schema_hashes.add(_schema_fingerprint(str(row[6])))
    if len(schema_hashes) != 1:
        raise ValueError("Bundle component feature schemas are inconsistent")


def _schema_fingerprint(path: str) -> str:
    schema_path = Path(path)
    if not schema_path.exists():
        return path
    digest = hashlib.sha256()
    with schema_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def activate_model(con, model_id: str) -> None:
    row = con.execute(
        """
        select model_type, feature_set_id, label_name, label_base, horizon_d
        from ml_model_registry
        where model_id = ?
        """,
        [model_id],
    ).fetchone()
    if row is None:
        raise ValueError(f"Unknown model_id: {model_id}")
    now = datetime.now(timezone.utc).isoformat()
    con.execute(
        """
        update ml_model_registry
        set is_active = false, deactivated_at = ?
        where model_type = ? and feature_set_id = ? and label_name = ? and label_base = ? and horizon_d = ?
        """,
        [now, *row],
    )
    con.execute(
        "update ml_model_registry set is_active = true, activated_at = ?, deactivated_at = null where model_id = ?",
        [now, model_id],
    )


def get_active_model(
    con,
    model_type: str,
    feature_set_id: str,
    label_name: str,
    label_base: str,
    horizon_d: int,
) -> dict[str, object]:
    row = con.execute(
        """
        select *
        from ml_model_registry
        where model_type = ? and feature_set_id = ? and label_name = ? and label_base = ?
          and horizon_d = ? and coalesce(is_active, false)
        """,
        [model_type, feature_set_id, label_name, label_base, horizon_d],
    ).fetchdf()
    if row.empty:
        raise ValueError("No active model found")
    return row.iloc[0].to_dict()
