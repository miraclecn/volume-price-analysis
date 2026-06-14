from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from dashboard.queries import (
    backtest_nav,
    continuous_nav,
    continuous_variant_options,
    data_health_summary,
    fold_metric_matrix,
    model_bundle_summary,
    run_dimensions,
    run_folds,
    run_metadata,
    selection_options,
    signal_preview,
)
from dashboard.ui import DEFAULT_DB, connect, get_streamlit


def main() -> None:
    st = get_streamlit()
    if st is None:
        print("VPA ML Dashboard: install streamlit to view dashboard UI")
        return

    st.set_page_config(page_title="VPA ML Dashboard", layout="wide")
    con = connect(DEFAULT_DB)
    try:
        render_run_dashboard(st, con)
    finally:
        con.close()


def render_run_dashboard(st, con) -> None:
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.25rem; padding-bottom: 2rem;}
        [data-testid="stMetric"] {border: 1px solid #d8dde6; border-radius: 6px; padding: 0.7rem 0.8rem; background: #fafbfc;}
        [data-testid="stSidebar"] {border-right: 1px solid #d8dde6;}
        h1, h2, h3 {letter-spacing: 0;}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("VPA ML Research Dashboard")
    st.caption(f"Read-only DuckDB source: {DEFAULT_DB}")

    runs = run_dimensions(con)
    if runs.empty:
        st.warning("No ml_runs found. Run a walk-forward or production train first, then refresh this dashboard.")
        health = data_health_summary(con)
        st.subheader("Data Health")
        st.json(health)
        return

    variants = continuous_variant_options(con, start_year=2020, end_year=2025)
    default_run = str(variants.iloc[0]["run_id"]) if not variants.empty else str(runs.iloc[0]["run_id"])
    run_ids = runs["run_id"].tolist()
    run_index = run_ids.index(default_run) if default_run in run_ids else 0
    selected_run = st.sidebar.selectbox("Run", run_ids, index=run_index)
    options = selection_options(con, selected_run)
    selected_fold = _optional_selectbox(st, "Fold", _option_values(options, "fold_id"))
    selected_strategy = _optional_selectbox(st, "Strategy", _option_values(options, "strategy_id"))
    selected_score = _optional_selectbox(st, "Score Version", _option_values(options, "score_version"))

    metadata = run_metadata(con, selected_run)
    metrics = fold_metric_matrix(
        con,
        run_id=selected_run,
        fold_id=selected_fold,
        strategy_id=selected_strategy,
        score_version=selected_score,
    )
    nav = backtest_nav(
        con,
        run_id=selected_run,
        fold_id=selected_fold,
        strategy_id=selected_strategy,
        score_version=selected_score,
    )

    _render_run_header(st, metadata, metrics)
    left, right = st.columns([1.5, 1])
    with left:
        st.subheader("NAV")
        if nav.empty:
            st.info("No NAV rows for the selected dimensions.")
        else:
            nav_chart = _series_frame(nav, "nav")
            st.line_chart(nav_chart)
    with right:
        st.subheader("Drawdown")
        drawdown = _drawdown_frame(nav)
        if drawdown.empty:
            st.info("No drawdown data for the selected dimensions.")
        else:
            st.line_chart(drawdown)

    st.subheader("Fold Metrics")
    if metrics.empty:
        st.info("No fold metrics for the selected dimensions.")
    else:
        chart_metrics = metrics.set_index("fold_id")[["annual_return", "max_drawdown"]]
        st.bar_chart(chart_metrics)
        st.dataframe(metrics, width="stretch", hide_index=True)

    _render_continuous_curve(st, con, selected_run, selected_strategy, selected_score)
    _render_lineage(st, con, selected_run)


def _render_run_header(st, metadata: pd.DataFrame, metrics: pd.DataFrame) -> None:
    meta = metadata.iloc[0].to_dict() if not metadata.empty else {}
    st.subheader(meta.get("experiment_name") or "Run Overview")
    cols = st.columns(5)
    cols[0].metric("Status", meta.get("status") or "")
    cols[1].metric("Folds", _metric_count(metrics, "fold_id"))
    cols[2].metric("Mean Return", _metric_mean(metrics, "annual_return", "{:.2%}"))
    cols[3].metric("Worst Drawdown", _metric_min(metrics, "max_drawdown", "{:.2%}"))
    cols[4].metric("Mean Calmar", _metric_mean(metrics, "calmar_like", "{:.2f}"))

    summary = {
        "run_id": meta.get("run_id", ""),
        "run_type": meta.get("run_type", ""),
        "feature_set_id": meta.get("feature_set_id", ""),
        "label_version": meta.get("label_version", ""),
        "score_version": meta.get("score_version", ""),
        "config_hash": meta.get("config_hash", ""),
        "git_commit": meta.get("git_commit", ""),
        "artifact_root": meta.get("artifact_root", ""),
    }
    st.dataframe(pd.DataFrame([summary]), width="stretch", hide_index=True)


def _render_lineage(st, con, run_id: str) -> None:
    folds = run_folds(con, run_id)
    bundles = model_bundle_summary(con)
    if not bundles.empty:
        bundles = bundles[bundles["run_id"] == run_id]
    signals = signal_preview(con)

    tab_folds, tab_bundles, tab_signals = st.tabs(["Folds", "Model Bundles", "Latest Signals"])
    with tab_folds:
        st.dataframe(folds, width="stretch", hide_index=True)
    with tab_bundles:
        st.dataframe(bundles, width="stretch", hide_index=True)
    with tab_signals:
        if not signals.empty and "source_bundle_id" in signals and not bundles.empty:
            bundle_ids = set(bundles["bundle_id"].dropna())
            signals = signals[signals["source_bundle_id"].isin(bundle_ids)]
        if not signals.empty and "source_sleeve" in signals:
            sleeve_weight = signals.groupby("source_sleeve", dropna=False)["target_weight"].sum().sort_values(ascending=False)
            st.bar_chart(sleeve_weight)
        st.dataframe(signals, width="stretch", hide_index=True)


def _render_continuous_curve(st, con, run_id: str, strategy_id: str | None, score_version: str | None) -> None:
    st.subheader("Continuous Curve")
    cols = st.columns([1, 1, 3])
    start_year = cols[0].number_input("Start Year", min_value=2000, max_value=2100, value=2020, step=1)
    end_year = cols[1].number_input("End Year", min_value=2000, max_value=2100, value=2025, step=1)
    variants = continuous_variant_options(con, run_id=run_id, start_year=int(start_year), end_year=int(end_year))
    fold_suffix = str(variants.iloc[0]["fold_suffix"]) if not variants.empty else None
    variant_score_version = str(variants.iloc[0]["score_version"]) if not variants.empty and pd.notna(variants.iloc[0]["score_version"]) else score_version
    stitched = continuous_nav(
        con,
        run_id=run_id,
        start_year=int(start_year),
        end_year=int(end_year),
        fold_suffix=fold_suffix,
        strategy_id=strategy_id,
        score_version=variant_score_version,
    )
    if stitched.empty:
        cols[2].info("No exact yearly folds found for the selected range. Expected fold IDs like wf_2020, wf_2021, ...")
        return

    stats = st.columns(4)
    total_return = stitched["continuous_nav"].iloc[-1] / stitched["continuous_nav"].iloc[0] - 1.0
    stats[0].metric("Total Return", f"{total_return:.2%}")
    stats[1].metric("Max Drawdown", f"{stitched['drawdown'].min():.2%}")
    stats[2].metric("Start NAV", f"{stitched['continuous_nav'].iloc[0]:,.0f}")
    stats[3].metric("End NAV", f"{stitched['continuous_nav'].iloc[-1]:,.0f}")
    st.caption(f"Fold variant: {fold_suffix or 'Base wf_YYYY'} | Score version: {variant_score_version or 'table default'}")

    left, right = st.columns([1.5, 1])
    with left:
        st.line_chart(stitched.set_index("sim_date")["continuous_nav"])
    with right:
        st.line_chart(stitched.set_index("sim_date")["drawdown"])

    annual = stitched.assign(year=stitched["sim_date"].str.slice(0, 4))
    annual_summary = annual.groupby("year", as_index=False).agg(
        fold_id=("fold_id", "first"),
        start_nav=("continuous_nav", "first"),
        end_nav=("continuous_nav", "last"),
        max_drawdown=("drawdown", "min"),
    )
    annual_summary["annual_return"] = annual_summary["end_nav"] / annual_summary["start_nav"] - 1.0
    st.dataframe(
        annual_summary[["year", "fold_id", "annual_return", "max_drawdown", "start_nav", "end_nav"]],
        width="stretch",
        hide_index=True,
    )


def _optional_selectbox(st, label: str, values: list[str]) -> str | None:
    choice = st.sidebar.selectbox(label, ["All", *values])
    return None if choice == "All" else choice


def _option_values(frame: pd.DataFrame, column: str) -> list[str]:
    if frame.empty or column not in frame:
        return []
    return sorted(str(value) for value in frame[column].dropna().unique() if str(value) != "")


def _series_frame(nav: pd.DataFrame, column: str) -> pd.DataFrame:
    if nav.empty or column not in nav:
        return pd.DataFrame()
    frame = nav.copy()
    frame["series"] = frame[["fold_id", "strategy_id", "score_version"]].fillna("").agg(" / ".join, axis=1)
    return frame.pivot_table(index="sim_date", columns="series", values=column, aggfunc="last").sort_index()


def _drawdown_frame(nav: pd.DataFrame) -> pd.DataFrame:
    series = _series_frame(nav, "nav")
    if series.empty:
        return series
    return series.divide(series.cummax()).subtract(1.0)


def _metric_count(frame: pd.DataFrame, column: str) -> int:
    if frame.empty or column not in frame:
        return 0
    return int(frame[column].dropna().nunique())


def _metric_mean(frame: pd.DataFrame, column: str, fmt: str) -> str:
    if frame.empty or column not in frame:
        return ""
    value = frame[column].dropna().mean()
    return "" if pd.isna(value) else fmt.format(value)


def _metric_min(frame: pd.DataFrame, column: str, fmt: str) -> str:
    if frame.empty or column not in frame:
        return ""
    value = frame[column].dropna().min()
    return "" if pd.isna(value) else fmt.format(value)


if __name__ == "__main__":
    main()
