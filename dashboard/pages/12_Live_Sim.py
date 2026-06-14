from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import duckdb
import pandas as pd

from dashboard.queries import live_sim_accounts, live_sim_nav, live_sim_order_summary
from dashboard.ui import get_streamlit


DEFAULT_LIVE_SIM_DB = "outputs/ml/live_sim/live_sim_state.duckdb"


def _connect_live_sim(db_path: str = DEFAULT_LIVE_SIM_DB):
    return duckdb.connect(str(Path(db_path)), read_only=True)


def _live_sim_table(con, table_name: str, account_id: str) -> pd.DataFrame:
    try:
        return con.execute(
            f"select * from {table_name} where account_id = ? order by 1, 2",
            [account_id],
        ).fetchdf()
    except (duckdb.CatalogException, duckdb.BinderException):
        return pd.DataFrame()


st = get_streamlit()
if st is None:
    print("Live Sim: install streamlit to view dashboard UI")
else:
    st.set_page_config(page_title="Live Sim", layout="wide")
    st.title("Live Sim")
    st.caption(f"Read-only DuckDB source: {DEFAULT_LIVE_SIM_DB}")

    db_path = Path(DEFAULT_LIVE_SIM_DB)
    if not db_path.exists():
        st.warning("Live simulation state database does not exist yet.")
    else:
        con = _connect_live_sim(DEFAULT_LIVE_SIM_DB)
        try:
            accounts = live_sim_accounts(con)
            if accounts.empty:
                st.info("No live_sim_account rows found.")
            else:
                account_id = st.sidebar.selectbox("Account", accounts["account_id"].tolist())
                nav = live_sim_nav(con, account_id)
                orders = live_sim_order_summary(con, account_id)

                if nav.empty:
                    st.info("No live_sim_nav rows found for this account.")
                else:
                    latest = nav.iloc[-1]
                    stats = st.columns(5)
                    stats[0].metric("NAV", f"{float(latest['nav']):,.0f}")
                    stats[1].metric("Total Return", f"{float(latest['total_return']):.2%}")
                    stats[2].metric("Current Drawdown", f"{float(latest['drawdown']):.2%}")
                    stats[3].metric("Max Drawdown", f"{float(nav['drawdown'].min()):.2%}")
                    stats[4].metric("Cash", f"{float(latest['cash']):,.0f}")

                    chart_frame = nav[nav["sim_date"] != "INITIAL"].copy()
                    left, right = st.columns([1.5, 1])
                    with left:
                        st.subheader("NAV")
                        st.line_chart(chart_frame.set_index("sim_date")["nav"])
                    with right:
                        st.subheader("Drawdown")
                        st.line_chart(chart_frame.set_index("sim_date")["drawdown"])

                    st.subheader("Daily Return")
                    st.bar_chart(chart_frame.set_index("sim_date")["daily_return"])
                    st.dataframe(nav, width="stretch", hide_index=True)

                st.subheader("Orders And Executions")
                if orders.empty:
                    st.info("No planned orders or executions found.")
                else:
                    st.dataframe(orders, width="stretch", hide_index=True)

                holdings, planned, executions = st.tabs(["Holdings", "Planned Orders", "Executions"])
                with holdings:
                    st.dataframe(_live_sim_table(con, "live_sim_holdings", account_id), width="stretch", hide_index=True)
                with planned:
                    st.dataframe(_live_sim_table(con, "live_sim_planned_orders", account_id), width="stretch", hide_index=True)
                with executions:
                    st.dataframe(_live_sim_table(con, "live_sim_executions", account_id), width="stretch", hide_index=True)
        finally:
            con.close()
