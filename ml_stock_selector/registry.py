from __future__ import annotations

from datetime import datetime, timezone
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
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    con.execute(
        """
        insert into ml_model_registry (
            model_id, model_type, feature_set_id, label_name, label_base, horizon_d,
            params_json, metrics_json, feature_schema_uri, artifact_uri,
            is_active, activated_at, deactivated_at, created_at, notes
        ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, false, null, null, ?, ?)
        on conflict (model_id) do update set
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
            params_json,
            metrics_json,
            feature_schema_uri,
            artifact_uri,
            now,
            notes,
        ],
    )


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

