from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dashboard.app import (
    _drawdown_frame,
    _optional_selectbox,
    _option_values,
    _series_frame,
)
from dashboard.queries import backtest_nav, fold_detail, fold_metric_matrix, run_dimensions, selection_options
from dashboard.ui import DEFAULT_DB, connect, get_streamlit


st = get_streamlit()
con = connect(DEFAULT_DB)
try:
    if st is None:
        print("Fold Detail: install streamlit to view dashboard UI")
    else:
        st.set_page_config(page_title="Fold Detail", layout="wide")
        st.title("Fold Detail")
        st.caption(f"Read-only DuckDB source: {DEFAULT_DB}")
        runs = run_dimensions(con)
        if runs.empty:
            st.info("No ml_runs found.")
        else:
            run_id = st.sidebar.selectbox("Run", runs["run_id"].tolist())
            options = selection_options(con, run_id)
            fold_id = _optional_selectbox(st, "Fold", _option_values(options, "fold_id"))
            strategy_id = _optional_selectbox(st, "Strategy", _option_values(options, "strategy_id"))
            score_version = _optional_selectbox(st, "Score Version", _option_values(options, "score_version"))
            nav = backtest_nav(con, run_id=run_id, fold_id=fold_id, strategy_id=strategy_id, score_version=score_version)
            metrics = fold_metric_matrix(con, run_id=run_id, fold_id=fold_id, strategy_id=strategy_id, score_version=score_version)
            left, right = st.columns(2)
            with left:
                st.subheader("NAV")
                st.line_chart(_series_frame(nav, "nav"))
            with right:
                st.subheader("Drawdown")
                st.line_chart(_drawdown_frame(nav))
            st.subheader("Metrics")
            st.dataframe(metrics, width="stretch", hide_index=True)
            detail = fold_detail(con, run_id=run_id, fold_id=fold_id)
            for name, frame in detail.items():
                st.subheader(name.title())
                st.dataframe(frame, width="stretch", hide_index=True)
finally:
    con.close()
