from __future__ import annotations

import os
from pathlib import Path

import duckdb
import pandas as pd


DEFAULT_DB = os.environ.get("VPA_DASHBOARD_ML_DB", "outputs/ml/ml_ret5_alpha_risk_20260619.duckdb")
DEFAULT_LIVE_SIM_DB = os.environ.get("VPA_DASHBOARD_LIVE_SIM_DB", "outputs/ml/live_sim/live_sim_state.duckdb")


def get_streamlit():
    try:
        import streamlit as st
    except ModuleNotFoundError:
        return None
    return st


def connect(db_path: str = DEFAULT_DB):
    return duckdb.connect(str(Path(db_path)), read_only=True)


def render_table(title: str, frame: pd.DataFrame, *, db_path: str = DEFAULT_DB) -> None:
    st = get_streamlit()
    if st is None:
        print(f"{title}: install streamlit to view dashboard UI")
        return
    st.set_page_config(page_title=title, layout="wide")
    st.title(title)
    st.caption(f"Read-only DuckDB source: {db_path}")
    st.dataframe(frame, width="stretch", hide_index=True)


def render_kv(title: str, values: dict[str, object], *, db_path: str = DEFAULT_DB) -> None:
    st = get_streamlit()
    if st is None:
        print(f"{title}: install streamlit to view dashboard UI")
        return
    st.set_page_config(page_title=title, layout="wide")
    st.title(title)
    st.caption(f"Read-only DuckDB source: {db_path}")
    cols = st.columns(min(4, max(1, len(values))))
    for idx, (key, value) in enumerate(values.items()):
        cols[idx % len(cols)].metric(key, value)
