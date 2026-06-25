from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import duckdb
import pandas as pd

from dashboard.queries import live_sim_accounts, live_sim_nav, live_sim_order_summary
from dashboard.strategy_data import (
    current_strategy_config,
    current_strategy_model_summary,
    current_strategy_report,
    strategy_report_paths,
)
from dashboard.ui import DEFAULT_DB, DEFAULT_LIVE_SIM_DB, get_streamlit


def main() -> None:
    st = get_streamlit()
    if st is None:
        print("VPA Dashboard: install streamlit to view dashboard UI")
        return

    st.set_page_config(page_title="VPA Strategy Dashboard", layout="wide")
    _style(st)
    render_dashboard(st)


def render_dashboard(st) -> None:
    config = current_strategy_config()
    report = current_strategy_report()
    models = current_strategy_model_summary()

    st.title("VPA Strategy Dashboard")
    st.caption(f"Read-only sources: ML={DEFAULT_DB} | live={DEFAULT_LIVE_SIM_DB}")

    account_id = _render_live_summary(st)
    st.divider()

    st.subheader("Current Strategy")
    summary = report.summary.iloc[0] if not report.summary.empty else pd.Series(dtype=object)
    cols = st.columns(6)
    cols[0].metric("Strategy", "mkt_tier_profit_protect")
    cols[1].metric("Total Return", _fmt_pct(summary.get("total_return")))
    cols[2].metric("Annual Return", _fmt_pct(summary.get("annual_return")))
    cols[3].metric("Max Drawdown", _fmt_pct(summary.get("max_drawdown")))
    cols[4].metric("Sharpe", _fmt_float(summary.get("sharpe")))
    cols[5].metric("Calmar", _fmt_float(summary.get("calmar")))

    selected_view = st.selectbox("Overview Section", ["Backtest Curve", "Risk Controls", "Model Snapshot"])
    if selected_view == "Backtest Curve":
        _render_backtest_snapshot(st, report)
    elif selected_view == "Risk Controls":
        _render_risk_controls(st, config)
    else:
        _render_model_snapshot(st, models)

    st.caption(
        f"Active account: {account_id or ''} | report dir: {strategy_report_paths()['report_dir']} | "
        "Dashboard is read-only and does not trigger training or orders."
    )


def _render_live_summary(st) -> str | None:
    st.subheader("Active Live Sim")
    db_path = Path(DEFAULT_LIVE_SIM_DB)
    if not db_path.exists():
        st.warning(f"Live sim state database not found: {DEFAULT_LIVE_SIM_DB}")
        return None

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        accounts = live_sim_accounts(con)
        if accounts.empty:
            st.info("No live simulation accounts found.")
            return None
        account_values = accounts["account_id"].tolist()
        default_index = account_values.index("profit_protect_paper") if "profit_protect_paper" in account_values else 0
        account_id = st.sidebar.selectbox("Live Account", account_values, index=default_index)
        nav = live_sim_nav(con, account_id)
        orders = live_sim_order_summary(con, account_id)
        nav = nav[nav["sim_date"] != "INITIAL"].copy() if not nav.empty else nav
        if nav.empty:
            st.info("Selected live simulation account has no NAV rows.")
            return account_id
        latest = nav.iloc[-1]
        position_count = _safe_scalar(
            con,
            "select count(*) from live_sim_holdings where account_id = ? and qty > 0",
            [account_id],
        )
        latest_plan_date = _safe_scalar(
            con,
            "select max(decision_date) from live_sim_planned_orders where account_id = ?",
            [account_id],
        )
        latest_plan_count = _safe_scalar(
            con,
            """
            select count(*)
            from live_sim_planned_orders
            where account_id = ?
              and status = 'planned'
              and decision_date = (
                  select max(decision_date)
                  from live_sim_planned_orders
                  where account_id = ?
              )
            """,
            [account_id, account_id],
        )
        cols = st.columns(6)
        cols[0].metric("Account", account_id)
        cols[1].metric("Latest Date", str(latest["sim_date"]))
        cols[2].metric("NAV", f"{float(latest['nav']):,.0f}")
        cols[3].metric("Return", _fmt_pct(latest["total_return"]))
        cols[4].metric("Drawdown", _fmt_pct(latest["drawdown"]))
        cols[5].metric("Positions / Latest Plans", f"{int(position_count or 0)} / {int(latest_plan_count or 0)}")
        st.caption(f"Latest plan date: {latest_plan_date or ''}")
        if not orders.empty:
            st.dataframe(orders.tail(5).sort_values("sim_date", ascending=False), width="stretch", hide_index=True)
        return account_id
    finally:
        con.close()


def _render_backtest_snapshot(st, report) -> None:
    if report.nav.empty:
        st.info("No continuous backtest NAV report found.")
        return
    left, right = st.columns([1.4, 1])
    with left:
        st.subheader("Backtest Curve")
        st.line_chart(report.nav.set_index("sim_date")["nav"])
    with right:
        st.subheader("Drawdown")
        st.line_chart(report.nav.set_index("sim_date")["drawdown"])

    st.subheader("Yearly Metrics")
    columns = [
        "year",
        "total_return",
        "max_drawdown",
        "avg_exposure",
        "trade_count",
        "win_rate",
        "loss_to_win",
        "profit_factor",
        "calmar",
    ]
    st.dataframe(_existing_columns(report.yearly, columns), width="stretch", hide_index=True)


def _render_risk_controls(st, config: dict[str, object]) -> None:
    constraints = dict(config.get("constraints") or {})
    holding_policy = dict(constraints.get("holding_policy") or {})
    execution = dict(config.get("execution") or {})
    rows = [
        ("target_positions", constraints.get("target_positions")),
        ("hard_max_positions", constraints.get("hard_max_positions")),
        ("max_initial_entries", constraints.get("max_initial_entries")),
        ("max_new_entries_per_day", constraints.get("max_new_entries_per_day")),
        ("min_adv20_amount", constraints.get("min_adv20_amount")),
        ("candidate_min_trade_score", constraints.get("candidate_min_trade_score")),
        ("candidate_absolute_min_rank_pct", constraints.get("candidate_absolute_min_rank_pct")),
        ("candidate_risk_max_rank_pct", constraints.get("candidate_risk_max_rank_pct")),
        ("core_risk_max_rank_pct", constraints.get("core_risk_max_rank_pct")),
        ("risk_exit_rank_pct", holding_policy.get("risk_exit_rank_pct")),
        ("risk_exit_prob", holding_policy.get("risk_exit_prob")),
        ("score_exit_threshold", holding_policy.get("sell_score_threshold")),
        ("target_hold_days", holding_policy.get("target_hold_days")),
        ("max_hold_days", holding_policy.get("max_hold_days")),
        ("profit_protect_min_days", config.get("profit_protect_min_days")),
        ("profit_protect_min_gain", config.get("profit_protect_min_gain")),
        ("profit_protect_exit_below", config.get("profit_protect_exit_below")),
        ("market_zero_below", config.get("market_zero_below")),
        ("market_half_below", config.get("market_half_below")),
        ("slippage_bps", execution.get("slippage_bps")),
        ("commission_bps", execution.get("commission_bps")),
        ("stamp_duty_bps", execution.get("stamp_duty_bps")),
    ]
    st.dataframe(pd.DataFrame(rows, columns=["parameter", "value"]), width="stretch", hide_index=True)


def _render_model_snapshot(st, models: pd.DataFrame) -> None:
    if models.empty:
        st.info("No fixed-round model manifests found.")
        return
    cols = st.columns(4)
    cols[0].metric("Run ID", str(models["run_id"].dropna().iloc[0]))
    cols[1].metric("Folds", str(models["fold_id"].nunique()))
    cols[2].metric("Alpha Rounds", "160")
    cols[3].metric("Risk Rounds", "120")
    st.dataframe(
        _existing_columns(
            models,
            [
                "fold_id",
                "model_role",
                "model_id",
                "objective",
                "n_estimators",
                "train_start",
                "train_end",
                "test_start",
                "test_end",
                "train_rows",
                "feature_set_id",
            ],
        ),
        width="stretch",
        hide_index=True,
    )


def _existing_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return frame[[column for column in columns if column in frame.columns]].copy()


def _safe_scalar(con, sql: str, params: list[object] | None = None) -> object | None:
    try:
        row = con.execute(sql, params or []).fetchone()
        return row[0] if row else None
    except (duckdb.CatalogException, duckdb.BinderException):
        return None


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


def _style(st) -> None:
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
        [data-testid="stMetric"] {border: 1px solid #d8dde6; border-radius: 6px; padding: 0.65rem 0.75rem; background: #fbfcfd;}
        [data-testid="stSidebar"] {border-right: 1px solid #d8dde6;}
        h1, h2, h3 {letter-spacing: 0;}
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
