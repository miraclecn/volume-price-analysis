from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dashboard.queries import fold_detail
from dashboard.ui import DEFAULT_DB, connect, get_streamlit


st = get_streamlit()
con = connect(DEFAULT_DB)
try:
    detail = fold_detail(con)
    if st is None:
        print("Fold Detail: install streamlit to view dashboard UI")
    else:
        st.set_page_config(page_title="Fold Detail", layout="wide")
        st.title("Fold Detail")
        st.caption(f"Read-only DuckDB source: {DEFAULT_DB}")
        for name, frame in detail.items():
            st.subheader(name.title())
            st.dataframe(frame, use_container_width=True, hide_index=True)
finally:
    con.close()
