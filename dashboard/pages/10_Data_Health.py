from __future__ import annotations

from dashboard.queries import data_health_summary
from dashboard.ui import DEFAULT_DB, connect, render_kv


con = connect(DEFAULT_DB)
try:
    render_kv("Data Health", data_health_summary(con), db_path=DEFAULT_DB)
finally:
    con.close()
