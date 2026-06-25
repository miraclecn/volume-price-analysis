from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import pickle
import sys

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml_stock_selector.feature_store_reader import FeatureStoreSpec, iter_feature_store_batches
from ml_stock_selector.storage import init_ml_db
from ml_stock_selector.universe import apply_universe_filter


@dataclass(frozen=True)
class RiskArtifactRef:
    fold_id: str
    model_id: str
    artifact_uri: Path
    feature_set_id: str
    horizon_d: int


@dataclass(frozen=True)
class SourceFold:
    fold_id: str
    start_date: str
    end_date: str
    row_count: int
    day_count: int


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ml-db", default="outputs/ml/ml_feature_variants_20260618.duckdb")
    parser.add_argument("--old-ml-db", default="outputs/ml/ml.duckdb")
    parser.add_argument("--feature-store-dir", default="outputs/ml/feature_store")
    parser.add_argument("--old-feature-store-version", default="v2_pv_only_001")
    parser.add_argument("--feature-set-id", default="vpa_d_sequence")
    parser.add_argument("--old-risk-run-id", default="wf_three_model_v2_adv10m_001")
    parser.add_argument("--source-run-id", default="wf_v2_fix12_drop_long_oldparams_20260618")
    parser.add_argument("--source-score-version", default="v2_three_model_drop_long_20260618")
    parser.add_argument("--output-run-id", default="wf_v2_fix12_drop_long_alpha_oldrisk_20260618")
    parser.add_argument("--output-score-version", default="v2_three_model_drop_long_alpha_oldrisk_20260618")
    parser.add_argument("--fold-id")
    parser.add_argument("--batch-size", type=int, default=100_000)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    risk_artifacts = _load_old_risk_artifacts(args.old_ml_db, args.old_risk_run_id)
    con = init_ml_db(args.ml_db)
    try:
        folds = _source_folds(con, args.source_run_id, args.source_score_version, args.fold_id)
        if not folds:
            raise ValueError("No source prediction folds matched the requested run/score_version")
        for fold in folds:
            risk_ref = risk_artifacts.get(fold.fold_id)
            if risk_ref is None:
                raise ValueError(f"No old risk model found for fold_id={fold.fold_id}")
            summary = _write_hybrid_fold(con, args, fold, risk_ref)
            print(
                "fold_id={fold_id} source_rows={source_rows} old_risk_rows={old_risk_rows} "
                "output_rows={output_rows} missing_old_risk={missing_old_risk} risk_model_id={risk_model_id}".format(
                    **summary
                )
            )
    finally:
        con.close()


def _load_old_risk_artifacts(old_ml_db: str, old_risk_run_id: str) -> dict[str, RiskArtifactRef]:
    con = duckdb.connect(old_ml_db, read_only=True)
    try:
        rows = con.execute(
            """
            select fold_id, model_id, artifact_uri, feature_set_id, horizon_d
            from ml_model_registry
            where model_type = 'risk_model'
              and run_id = ?
              and fold_id is not null
            order by fold_id
            """,
            [old_risk_run_id],
        ).fetchdf()
    finally:
        con.close()
    out: dict[str, RiskArtifactRef] = {}
    for row in rows.itertuples(index=False):
        artifact_uri = Path(str(row.artifact_uri))
        if not artifact_uri.exists():
            raise FileNotFoundError(f"Old risk artifact is missing: {artifact_uri}")
        out[str(row.fold_id)] = RiskArtifactRef(
            fold_id=str(row.fold_id),
            model_id=str(row.model_id),
            artifact_uri=artifact_uri,
            feature_set_id=str(row.feature_set_id),
            horizon_d=int(row.horizon_d),
        )
    return out


def _source_folds(
    con: duckdb.DuckDBPyConnection,
    source_run_id: str,
    source_score_version: str,
    fold_id: str | None,
) -> list[SourceFold]:
    where = ["run_id = ?", "score_version = ?"]
    params: list[object] = [source_run_id, source_score_version]
    if fold_id:
        where.append("fold_id = ?")
        params.append(fold_id)
    rows = con.execute(
        f"""
        select
            fold_id,
            min(trade_date) as start_date,
            max(trade_date) as end_date,
            count(*) as row_count,
            count(distinct trade_date) as day_count
        from ml_predictions_daily
        where {" and ".join(where)}
        group by fold_id
        order by fold_id
        """,
        params,
    ).fetchdf()
    return [
        SourceFold(
            fold_id=str(row.fold_id),
            start_date=str(row.start_date),
            end_date=str(row.end_date),
            row_count=int(row.row_count),
            day_count=int(row.day_count),
        )
        for row in rows.itertuples(index=False)
    ]


def _write_hybrid_fold(
    con: duckdb.DuckDBPyConnection,
    args: argparse.Namespace,
    fold: SourceFold,
    risk_ref: RiskArtifactRef,
) -> dict[str, object]:
    with risk_ref.artifact_uri.open("rb") as handle:
        risk_model = pickle.load(handle)
    feature_columns = list(getattr(risk_model, "feature_columns", []))
    if not feature_columns:
        raise ValueError(f"Old risk model has no feature_columns: {risk_ref.model_id}")

    con.execute("drop table if exists old_risk_raw")
    con.execute(
        """
        create temporary table old_risk_raw (
            trade_date varchar not null,
            code varchar not null,
            risk_prob double not null
        )
        """
    )
    old_risk_rows = _score_old_risk_to_temp(con, args, fold, risk_model, feature_columns)
    con.execute("drop table if exists old_risk_ranked")
    con.execute(
        """
        create temporary table old_risk_ranked as
        select
            trade_date,
            code,
            risk_prob,
            percent_rank() over (partition by trade_date order by risk_prob) as risk_rank_pct
        from old_risk_raw
        """
    )
    missing_old_risk = con.execute(
        """
        select count(*)
        from ml_predictions_daily p
        left join old_risk_ranked r
          on p.trade_date = r.trade_date and p.code = r.code
        where p.run_id = ?
          and p.fold_id = ?
          and p.score_version = ?
          and r.code is null
        """,
        [args.source_run_id, fold.fold_id, args.source_score_version],
    ).fetchone()[0]

    con.execute(
        """
        delete from ml_predictions_daily
        where run_id = ? and fold_id = ? and score_version = ?
        """,
        [args.output_run_id, fold.fold_id, args.output_score_version],
    )
    generated_at = datetime.now(timezone.utc).isoformat()
    con.execute(
        """
        insert into ml_predictions_daily (
            trade_date, code, model_id, horizon_d,
            alpha_score, alpha_rank_pct,
            absolute_score, absolute_rank_pct,
            reg_score,
            active_score, active_rank_pct,
            risk_score, risk_prob, risk_rank_pct,
            context_score, liquidity_score, relative_strength_pct, resonance_pct, penalty_score,
            core_score, trade_score, trade_score_v2,
            score_version, run_id, fold_id,
            absolute_model_id, active_model_id, risk_model_id,
            feature_set_id, generated_at
        )
        select
            p.trade_date,
            p.code,
            'three_model:' || p.absolute_model_id || ':' || p.active_model_id || ':' || ? as model_id,
            p.horizon_d,
            p.absolute_score as alpha_score,
            p.absolute_rank_pct as alpha_rank_pct,
            p.absolute_score,
            p.absolute_rank_pct,
            p.reg_score,
            p.active_score,
            coalesce(p.active_rank_pct, p.absolute_rank_pct) as active_rank_pct,
            coalesce(r.risk_prob, 1.0) as risk_score,
            coalesce(r.risk_prob, 1.0) as risk_prob,
            coalesce(r.risk_rank_pct, 1.0) as risk_rank_pct,
            p.context_score,
            p.liquidity_score,
            coalesce(p.relative_strength_pct, 0.5) as relative_strength_pct,
            coalesce(p.resonance_pct, 0.5) as resonance_pct,
            coalesce(p.penalty_score, 0.0) as penalty_score,
            0.55 * p.absolute_rank_pct
                + 0.35 * coalesce(p.active_rank_pct, p.absolute_rank_pct)
                - 0.25 * coalesce(r.risk_rank_pct, 1.0) as core_score,
            null as trade_score,
            0.55 * p.absolute_rank_pct
                + 0.35 * coalesce(p.active_rank_pct, p.absolute_rank_pct)
                - 0.25 * coalesce(r.risk_rank_pct, 1.0) as trade_score_v2,
            ? as score_version,
            ? as run_id,
            p.fold_id,
            p.absolute_model_id,
            p.active_model_id,
            ? as risk_model_id,
            p.feature_set_id,
            ? as generated_at
        from ml_predictions_daily p
        left join old_risk_ranked r
          on p.trade_date = r.trade_date and p.code = r.code
        where p.run_id = ?
          and p.fold_id = ?
          and p.score_version = ?
        """,
        [
            risk_ref.model_id,
            args.output_score_version,
            args.output_run_id,
            risk_ref.model_id,
            generated_at,
            args.source_run_id,
            fold.fold_id,
            args.source_score_version,
        ],
    )
    output_rows = con.execute(
        """
        select count(*)
        from ml_predictions_daily
        where run_id = ? and fold_id = ? and score_version = ?
        """,
        [args.output_run_id, fold.fold_id, args.output_score_version],
    ).fetchone()[0]
    con.execute("drop table if exists old_risk_ranked")
    con.execute("drop table if exists old_risk_raw")
    return {
        "fold_id": fold.fold_id,
        "source_rows": fold.row_count,
        "old_risk_rows": int(old_risk_rows),
        "output_rows": int(output_rows),
        "missing_old_risk": int(missing_old_risk),
        "risk_model_id": risk_ref.model_id,
    }


def _score_old_risk_to_temp(
    con: duckdb.DuckDBPyConnection,
    args: argparse.Namespace,
    fold: SourceFold,
    risk_model: object,
    feature_columns: list[str],
) -> int:
    spec = FeatureStoreSpec(
        feature_store_dir=args.feature_store_dir,
        dataset_version=args.old_feature_store_version,
        feature_set_id=args.feature_set_id,
    )
    rows = 0
    columns = ["trade_date", "code", *feature_columns]
    for features in iter_feature_store_batches(
        spec,
        fold.start_date,
        fold.end_date,
        columns=columns,
        batch_size=args.batch_size,
    ):
        features = apply_universe_filter(features, exclude_bse=True)
        if features.empty:
            continue
        matrix = (
            features.reindex(columns=feature_columns, fill_value=0.0)
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
            .astype("float32")
        )
        raw = features[["trade_date", "code"]].copy()
        raw["trade_date"] = raw["trade_date"].astype(str)
        raw["code"] = raw["code"].astype(str)
        raw["risk_prob"] = list(risk_model.predict_proba_matrix(matrix))
        con.register("_old_risk_chunk", raw)
        try:
            con.execute("insert into old_risk_raw select trade_date, code, risk_prob from _old_risk_chunk")
        finally:
            con.unregister("_old_risk_chunk")
        rows += len(raw)
    return rows


if __name__ == "__main__":
    main()
