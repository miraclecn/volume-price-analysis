from __future__ import annotations

from dashboard.queries import fixed_horizon_compare
from dashboard.ui import DEFAULT_DB, connect, render_table


con = connect(DEFAULT_DB)
try:
    render_table("Fixed Horizon Compare", fixed_horizon_compare(con), db_path=DEFAULT_DB)
finally:
    con.close()

