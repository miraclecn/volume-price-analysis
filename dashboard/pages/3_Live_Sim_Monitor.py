from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import duckdb
import pandas as pd

from dashboard.queries import live_sim_accounts, live_sim_nav, live_sim_order_summary
from dashboard.ui import DEFAULT_LIVE_SIM_DB, get_streamlit


def main() -> None:
    st = get_streamlit()
    if st is None:
        print("Live Sim Monitor: install streamlit to view dashboard UI")
        return

    st.set_page_config(page_title="Live Sim Monitor", layout="wide")
    st.title("Live Sim Monitor")
    st.caption(f"Read-only live simulation source: {DEFAULT_LIVE_SIM_DB}")

    db_path = Path(DEFAULT_LIVE_SIM_DB)
    if not db_path.exists():
        st.warning("Live simulation state database does not exist yet.")
        return

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        accounts = live_sim_accounts(con)
        if accounts.empty:
            st.info("No live simulation accounts found.")
            return
        account_values = accounts["account_id"].tolist()
        default_index = account_values.index("profit_protect_paper") if "profit_protect_paper" in account_values else 0
        account_id = st.sidebar.selectbox("Account", account_values, index=default_index)

        nav = live_sim_nav(con, account_id)
        orders = live_sim_order_summary(con, account_id)
        holdings = _live_sim_table(con, "live_sim_holdings", account_id)
        planned = _live_sim_table(con, "live_sim_planned_orders", account_id)
        executions = _live_sim_table(con, "live_sim_executions", account_id)
        path_stats = _live_sim_table(con, "live_sim_holding_path_stats", account_id)

        usable_nav = nav[nav["sim_date"] != "INITIAL"].copy() if not nav.empty else nav
        if usable_nav.empty:
            st.info("No NAV rows found for this account.")
        else:
            latest = usable_nav.iloc[-1]
            latest_plan_date = None if planned.empty else planned["decision_date"].max()
            latest_planned = planned[planned["decision_date"] == latest_plan_date] if latest_plan_date else planned
            latest_plan_count = int((latest_planned["status"].astype(str) == "planned").sum()) if not latest_planned.empty else 0
            cols = st.columns(6)
            cols[0].metric("NAV", f"{float(latest['nav']):,.0f}")
            cols[1].metric("Total Return", _fmt_pct(latest["total_return"]))
            cols[2].metric("Drawdown", _fmt_pct(latest["drawdown"]))
            cols[3].metric("Cash", f"{float(latest['cash']):,.0f}")
            cols[4].metric("Positions", str(_positive_position_count(holdings)))
            cols[5].metric("Latest Plans", str(latest_plan_count))
            st.caption(f"Latest plan date: {latest_plan_date or ''}")

        tab_curve, tab_activity, tab_holdings, tab_planned, tab_exec, tab_path = st.tabs(
            ["Curve", "Order Activity", "Holdings", "Planned Orders", "Executions", "Holding Path"]
        )
        with tab_curve:
            left, right = st.columns([1.4, 1])
            with left:
                st.subheader("NAV")
                if not usable_nav.empty:
                    st.line_chart(usable_nav.set_index("sim_date")["nav"])
            with right:
                st.subheader("Drawdown")
                if not usable_nav.empty:
                    st.line_chart(usable_nav.set_index("sim_date")["drawdown"])
        with tab_activity:
            st.subheader("Order Activity")
            st.dataframe(orders.sort_values("sim_date", ascending=False), width="stretch", hide_index=True)
        with tab_holdings:
            st.dataframe(holdings, width="stretch", hide_index=True)
        with tab_planned:
            st.dataframe(planned.sort_values(["decision_date", "code"], ascending=[False, True]), width="stretch", hide_index=True)
        with tab_exec:
            st.dataframe(executions.sort_values(["sim_date", "code"], ascending=[False, True]), width="stretch", hide_index=True)
        with tab_path:
            st.dataframe(path_stats, width="stretch", hide_index=True)
    finally:
        con.close()


def _live_sim_table(con, table_name: str, account_id: str) -> pd.DataFrame:
    try:
        return con.execute(
            f"select * from {table_name} where account_id = ? order by 1, 2",
            [account_id],
        ).fetchdf()
    except (duckdb.CatalogException, duckdb.BinderException):
        return pd.DataFrame()


def _positive_position_count(holdings: pd.DataFrame) -> int:
    if holdings.empty:
        return 0
    if "qty" not in holdings:
        return len(holdings)
    return int((pd.to_numeric(holdings["qty"], errors="coerce").fillna(0) > 0).sum())


def _fmt_pct(value: object) -> str:
    try:
        return f"{float(value):.2%}"
    except (TypeError, ValueError):
        return ""


if __name__ == "__main__":
    main()
