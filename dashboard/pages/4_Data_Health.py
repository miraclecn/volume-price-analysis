from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import duckdb
import pandas as pd

from dashboard.queries import data_health_summary
from dashboard.strategy_data import strategy_report_paths
from dashboard.ui import DEFAULT_DB, DEFAULT_LIVE_SIM_DB, connect, get_streamlit


def main() -> None:
    st = get_streamlit()
    if st is None:
        print("Data Health: install streamlit to view dashboard UI")
        return

    st.set_page_config(page_title="Data Health", layout="wide")
    st.title("Data Health")
    st.caption("Read-only status for ML DB, live sim DB, and current strategy reports.")

    tab_ml, tab_live, tab_reports = st.tabs(["ML DB", "Live Sim DB", "Strategy Reports"])
    with tab_ml:
        st.subheader("ML DB")
        st.caption(DEFAULT_DB)
        if Path(DEFAULT_DB).exists():
            con = connect(DEFAULT_DB)
            try:
                st.json(data_health_summary(con))
                st.dataframe(_table_counts(con), width="stretch", hide_index=True)
            finally:
                con.close()
        else:
            st.warning("ML DB not found.")

    with tab_live:
        st.subheader("Live Sim DB")
        st.caption(DEFAULT_LIVE_SIM_DB)
        if Path(DEFAULT_LIVE_SIM_DB).exists():
            con = duckdb.connect(DEFAULT_LIVE_SIM_DB, read_only=True)
            try:
                st.dataframe(_table_counts(con), width="stretch", hide_index=True)
            finally:
                con.close()
        else:
            st.warning("Live sim DB not found.")

    with tab_reports:
        st.subheader("Strategy Reports")
        rows = []
        for name, path in strategy_report_paths().items():
            if name == "report_dir":
                continue
            rows.append(
                {
                    "artifact": name,
                    "path": str(path),
                    "exists": path.exists(),
                    "size_mb": round(path.stat().st_size / 1024 / 1024, 3) if path.exists() else None,
                    "modified_at": pd.to_datetime(path.stat().st_mtime, unit="s") if path.exists() else None,
                }
            )
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def _table_counts(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    tables = con.execute(
        "select table_name from information_schema.tables where table_schema = 'main' order by table_name"
    ).fetchdf()
    rows = []
    for table_name in tables["table_name"].tolist():
        if not str(table_name).startswith(("ml_", "live_")):
            continue
        try:
            count = con.execute(f"select count(*) from {table_name}").fetchone()[0]
        except duckdb.Error:
            count = None
        rows.append({"table_name": table_name, "row_count": count})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
