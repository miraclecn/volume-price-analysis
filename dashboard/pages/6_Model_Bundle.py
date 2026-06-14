from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dashboard.queries import model_bundle_summary
from dashboard.ui import DEFAULT_DB, connect, render_table


con = connect(DEFAULT_DB)
try:
    render_table("Model Bundle", model_bundle_summary(con), db_path=DEFAULT_DB)
finally:
    con.close()
