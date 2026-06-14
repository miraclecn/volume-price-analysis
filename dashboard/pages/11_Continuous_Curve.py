from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from dashboard.queries import continuous_nav, continuous_variant_options, run_dimensions, selection_options
from dashboard.ui import DEFAULT_DB, connect, get_streamlit


def _option_values(frame: pd.DataFrame, column: str) -> list[str]:
    if frame.empty or column not in frame:
        return []
    return sorted(str(value) for value in frame[column].dropna().unique() if str(value) != "")


def _optional_selectbox(st, label: str, values: list[str]) -> str | None:
    choice = st.sidebar.selectbox(label, ["All", *values])
    return None if choice == "All" else choice


def _variant_labels(variants: pd.DataFrame) -> list[str]:
    if variants.empty:
        return []
    labels = []
    for row in variants.itertuples(index=False):
        labels.append(
            f"{row.fold_suffix} | min {row.min_annual_return:.1%} | geom {row.geometric_annual_return:.1%}"
        )
    return labels


st = get_streamlit()
con = connect(DEFAULT_DB)
try:
    if st is None:
        print("Continuous Curve: install streamlit to view dashboard UI")
    else:
        st.set_page_config(page_title="Continuous Curve", layout="wide")
        st.title("Continuous Curve")
        st.caption(f"Read-only DuckDB source: {DEFAULT_DB}")

        runs = run_dimensions(con)
        if runs.empty:
            st.info("No run_id found in ml_runs or legacy backtest tables.")
        else:
            start_year = st.sidebar.number_input("Start Year", min_value=2000, max_value=2100, value=2020, step=1)
            end_year = st.sidebar.number_input("End Year", min_value=2000, max_value=2100, value=2025, step=1)
            variants = continuous_variant_options(con, start_year=int(start_year), end_year=int(end_year))
            default_run = str(variants.iloc[0]["run_id"]) if not variants.empty else str(runs.iloc[0]["run_id"])
            run_ids = runs["run_id"].tolist()
            run_index = run_ids.index(default_run) if default_run in run_ids else 0
            run_id = st.sidebar.selectbox("Run", run_ids, index=run_index)
            run_variants = continuous_variant_options(con, run_id=run_id, start_year=int(start_year), end_year=int(end_year))
            variant_labels = _variant_labels(run_variants)
            variant_choice = st.sidebar.selectbox("Fold Variant", ["Base wf_YYYY", *variant_labels])
            fold_suffix = None
            score_version = None
            if variant_choice != "Base wf_YYYY" and not run_variants.empty:
                selected_variant = run_variants.iloc[variant_labels.index(variant_choice)]
                fold_suffix = str(selected_variant["fold_suffix"])
                score_version = str(selected_variant["score_version"]) if pd.notna(selected_variant["score_version"]) else None
            options = selection_options(con, run_id)
            strategy_id = _optional_selectbox(st, "Strategy", _option_values(options, "strategy_id"))

            curve = continuous_nav(
                con,
                run_id=run_id,
                start_year=int(start_year),
                end_year=int(end_year),
                fold_suffix=fold_suffix,
                strategy_id=strategy_id,
                score_version=score_version,
            )
            if curve.empty:
                st.warning("No exact yearly fold NAV found. Expected fold IDs like wf_2020, wf_2021, ...")
            else:
                total_return = curve["continuous_nav"].iloc[-1] / curve["continuous_nav"].iloc[0] - 1.0
                stats = st.columns(5)
                stats[0].metric("Run", run_id)
                stats[1].metric("Rows", f"{len(curve):,}")
                stats[2].metric("Total Return", f"{total_return:.2%}")
                stats[3].metric("Max Drawdown", f"{curve['drawdown'].min():.2%}")
                stats[4].metric("End NAV", f"{curve['continuous_nav'].iloc[-1]:,.0f}")
                st.caption(f"Fold variant: {fold_suffix or 'Base wf_YYYY'} | Score version: {score_version or 'table default'}")

                left, right = st.columns([1.5, 1])
                with left:
                    st.subheader("Continuous NAV")
                    st.line_chart(curve.set_index("sim_date")["continuous_nav"])
                with right:
                    st.subheader("Continuous Drawdown")
                    st.line_chart(curve.set_index("sim_date")["drawdown"])

                annual = curve.assign(year=curve["sim_date"].str.slice(0, 4))
                annual_summary = annual.groupby("year", as_index=False).agg(
                    fold_id=("fold_id", "first"),
                    start_nav=("continuous_nav", "first"),
                    end_nav=("continuous_nav", "last"),
                    max_drawdown=("drawdown", "min"),
                )
                annual_summary["annual_return"] = annual_summary["end_nav"] / annual_summary["start_nav"] - 1.0
                st.subheader("Yearly Stitch Summary")
                st.dataframe(
                    annual_summary[["year", "fold_id", "annual_return", "max_drawdown", "start_nav", "end_nav"]],
                    width="stretch",
                    hide_index=True,
                )
finally:
    con.close()
