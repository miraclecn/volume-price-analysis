from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from dashboard.strategy_data import current_strategy_report, strategy_report_paths
from dashboard.ui import get_streamlit


def main() -> None:
    st = get_streamlit()
    if st is None:
        print("Strategy Research: install streamlit to view dashboard UI")
        return

    st.set_page_config(page_title="Strategy Research", layout="wide")
    st.title("Strategy Research")
    st.caption(f"Read-only report source: {strategy_report_paths()['report_dir']}")

    report = current_strategy_report()
    if report.summary.empty:
        st.warning("No strategy summary report found.")
        return

    summary = report.summary.iloc[0]
    cols = st.columns(7)
    cols[0].metric("Variant", str(summary.get("variant", "")))
    cols[1].metric("Total Return", _fmt_pct(summary.get("total_return")))
    cols[2].metric("Annual Return", _fmt_pct(summary.get("annual_return")))
    cols[3].metric("Max Drawdown", _fmt_pct(summary.get("max_drawdown")))
    cols[4].metric("Sharpe", _fmt_float(summary.get("sharpe")))
    cols[5].metric("Sortino", _fmt_float(summary.get("sortino")))
    cols[6].metric("Profit Factor", _fmt_float(summary.get("profit_factor")))

    tab_curve, tab_yearly, tab_exit, tab_orders, tab_diagnostics = st.tabs(
        ["Curve", "Yearly", "Exit Attribution", "Orders", "Diagnostics"]
    )
    with tab_curve:
        left, right = st.columns([1.4, 1])
        with left:
            st.subheader("Backtest Curve")
            if not report.nav.empty:
                st.line_chart(report.nav.set_index("sim_date")["nav"])
        with right:
            st.subheader("Drawdown")
            if not report.nav.empty:
                st.line_chart(report.nav.set_index("sim_date")["drawdown"])

    with tab_yearly:
        st.subheader("Yearly Metrics")
        st.dataframe(
            _existing_columns(
                report.yearly,
                [
                    "year",
                    "total_return",
                    "max_drawdown",
                    "avg_exposure",
                    "trade_count",
                    "win_rate",
                    "avg_win",
                    "avg_loss",
                    "loss_to_win",
                    "profit_factor",
                    "calmar",
                ],
            ),
            width="stretch",
            hide_index=True,
        )

    with tab_exit:
        st.subheader("Exit Attribution")
        st.dataframe(report.exit_attribution, width="stretch", hide_index=True)
        if not report.exit_attribution.empty and "total_realized_pnl" in report.exit_attribution:
            st.bar_chart(report.exit_attribution.set_index("exit_reason")["total_realized_pnl"])

    with tab_orders:
        st.subheader("Filled Orders")
        frame = report.orders.sort_values("sim_date", ascending=False) if not report.orders.empty else pd.DataFrame()
        st.dataframe(frame.head(5000), width="stretch", hide_index=True)

    with tab_diagnostics:
        st.subheader("Construction Diagnostics")
        frame = (
            report.diagnostics.sort_values("trade_date", ascending=False)
            if not report.diagnostics.empty
            else pd.DataFrame()
        )
        st.dataframe(frame.head(5000), width="stretch", hide_index=True)


def _existing_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return frame[[column for column in columns if column in frame.columns]].copy()


def _fmt_pct(value: object) -> str:
    try:
        return f"{float(value):.2%}"
    except (TypeError, ValueError):
        return ""


def _fmt_float(value: object) -> str:
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return ""


if __name__ == "__main__":
    main()
