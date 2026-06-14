from __future__ import annotations

from dashboard.queries import live_monitor
from dashboard.ui import DEFAULT_DB, connect, render_table


con = connect(DEFAULT_DB)
try:
    render_table("Live Monitor", live_monitor(con), db_path=DEFAULT_DB)
finally:
    con.close()

