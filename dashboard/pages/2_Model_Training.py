from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from dashboard.strategy_data import current_strategy_model_summary
from dashboard.ui import get_streamlit


def main() -> None:
    st = get_streamlit()
    if st is None:
        print("Model Training: install streamlit to view dashboard UI")
        return

    st.set_page_config(page_title="Model Training", layout="wide")
    st.title("Model Training")
    st.caption("Current strategy models: fixed_round_full_history, alpha=160 rounds, risk=120 rounds.")

    models = current_strategy_model_summary()
    if models.empty:
        st.warning("No fixed-round model manifests found.")
        return

    roles = ["All", *sorted(models["model_role"].dropna().unique().tolist())]
    selected_role = st.sidebar.selectbox("Model Role", roles)
    frame = models if selected_role == "All" else models[models["model_role"] == selected_role]

    cols = st.columns(5)
    cols[0].metric("Run ID", str(models["run_id"].dropna().iloc[0]))
    cols[1].metric("Folds", str(models["fold_id"].nunique()))
    cols[2].metric("Feature Set", str(models["feature_set_id"].dropna().iloc[0]))
    cols[3].metric("Alpha Rounds", "160")
    cols[4].metric("Risk Rounds", "120")

    tab_windows, tab_params, tab_artifacts = st.tabs(["Training Windows", "Model Parameters", "Artifacts"])
    with tab_windows:
        st.subheader("Walk-forward Training Windows")
        st.dataframe(
            _existing_columns(
                frame,
                [
                    "fold_id",
                    "model_role",
                    "train_start",
                    "train_end",
                    "valid_start",
                    "valid_end",
                    "test_start",
                    "test_end",
                    "train_rows",
                    "train_window_mode",
                    "source_train_window_mode",
                    "model_mode",
                ],
            ),
            width="stretch",
            hide_index=True,
        )

    with tab_params:
        st.subheader("Model Parameters")
        st.dataframe(
            _existing_columns(
                frame,
                [
                    "fold_id",
                    "model_role",
                    "objective",
                    "metric",
                    "n_estimators",
                    "learning_rate",
                    "num_leaves",
                    "min_data_in_leaf",
                    "early_stopping_rounds",
                    "alpha_eval_metric",
                    "alpha_eval_target",
                    "label_name",
                    "label_base",
                    "horizon_d",
                ],
            ),
            width="stretch",
            hide_index=True,
        )

    with tab_artifacts:
        st.subheader("Model Artifacts")
        st.dataframe(
            _existing_columns(
                frame,
                [
                    "fold_id",
                    "model_role",
                    "model_id",
                    "artifact_uri",
                    "feature_schema_uri",
                    "manifest_path",
                ],
            ),
            width="stretch",
            hide_index=True,
        )


def _existing_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return frame[[column for column in columns if column in frame.columns]].copy()


if __name__ == "__main__":
    main()
