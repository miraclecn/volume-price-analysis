from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


FUNDAMENTAL_FEATURE_COLUMNS: list[tuple[str, str]] = [
    ("eps", "fund_eps"),
    ("roe", "fund_roe"),
    ("roa", "fund_roa"),
    ("gross_margin", "fund_gross_margin"),
    ("netprofit_margin", "fund_netprofit_margin"),
    ("current_ratio", "fund_current_ratio"),
    ("debt_to_assets", "fund_debt_to_assets"),
    ("revenue_ps", "fund_revenue_ps"),
    ("netprofit_yoy", "fund_netprofit_yoy"),
    ("dt_netprofit_yoy", "fund_dt_netprofit_yoy"),
    ("or_yoy", "fund_or_yoy"),
    ("q_sales_yoy", "fund_q_sales_yoy"),
    ("assets_yoy", "fund_assets_yoy"),
    ("equity_yoy", "fund_equity_yoy"),
]


def load_fundamental_features_for_metadata(raw_db_path: str | Path, metadata: pd.DataFrame) -> pd.DataFrame:
    """Return raw daily fundamental features aligned to matrix metadata rows."""
    if "trade_date" not in metadata or "code" not in metadata:
        raise ValueError("metadata must contain trade_date and code columns")

    columns = [target for _, target in FUNDAMENTAL_FEATURE_COLUMNS]
    if metadata.empty:
        return pd.DataFrame(columns=columns, dtype=np.float32)

    meta = pd.DataFrame(
        {
            "_row_id": np.arange(len(metadata), dtype=np.int64),
            "code": metadata["code"].astype(str).to_numpy(),
            "date": _date_key_series(metadata["trade_date"]).to_numpy(),
        }
    )
    con = duckdb.connect(str(raw_db_path), read_only=True)
    try:
        con.register("_fundamental_metadata", meta)
        select_list = ",\n                ".join(
            f"try_cast(f.{source} as double) as {target}" for source, target in FUNDAMENTAL_FEATURE_COLUMNS
        )
        frame = con.execute(
            f"""
            select
                m._row_id,
                {select_list}
            from _fundamental_metadata m
            left join mart_fundamental f
              on m.code = f.code
             and m.date = f.date
            order by m._row_id
            """
        ).fetchdf()
    finally:
        con.close()

    features = frame[columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return features.astype(np.float32).reset_index(drop=True)


def append_fundamental_columns(
    matrix: np.ndarray,
    metadata: pd.DataFrame,
    raw_db_path: str | Path,
) -> tuple[np.ndarray, list[str]]:
    features = load_fundamental_features_for_metadata(raw_db_path, metadata)
    base = np.asarray(matrix, dtype=np.float32)
    augmented = np.concatenate([base, features.to_numpy(dtype=np.float32, copy=False)], axis=1)
    return augmented.astype(np.float32, copy=False), list(features.columns)


def _date_key_series(values: pd.Series) -> pd.Series:
    text = values.astype(str)
    compact = text.str.replace("-", "", regex=False)
    needs_parse = ~compact.str.fullmatch(r"\d{8}")
    if needs_parse.any():
        compact.loc[needs_parse] = pd.to_datetime(values.loc[needs_parse], errors="coerce").dt.strftime("%Y%m%d")
    return compact.fillna("")
