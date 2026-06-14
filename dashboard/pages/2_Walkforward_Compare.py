from __future__ import annotations

from dashboard.queries import walkforward_compare
from dashboard.ui import DEFAULT_DB, connect, render_table


con = connect(DEFAULT_DB)
try:
    render_table("Walk-forward Compare", walkforward_compare(con), db_path=DEFAULT_DB)
finally:
    con.close()

