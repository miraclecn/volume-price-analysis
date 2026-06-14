from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dashboard.queries import run_registry
from dashboard.ui import DEFAULT_DB, connect, render_table


def main() -> None:
    con = connect(DEFAULT_DB)
    try:
        render_table("Run Registry", run_registry(con), db_path=DEFAULT_DB)
    finally:
        con.close()


if __name__ == "__main__":
    main()
