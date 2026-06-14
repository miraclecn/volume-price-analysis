from __future__ import annotations

from dashboard.queries import score_mode_compare
from dashboard.ui import DEFAULT_DB, connect, render_table


con = connect(DEFAULT_DB)
try:
    render_table("Score Mode Compare", score_mode_compare(con), db_path=DEFAULT_DB)
finally:
    con.close()

