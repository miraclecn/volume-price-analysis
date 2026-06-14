from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dashboard.queries import portfolio_diagnostics
from dashboard.ui import DEFAULT_DB, connect, render_table


con = connect(DEFAULT_DB)
try:
    render_table("Portfolio Diagnostics", portfolio_diagnostics(con), db_path=DEFAULT_DB)
finally:
    con.close()
