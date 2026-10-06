import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
import matplotlib.pyplot as plt

from pathlib import Path


# ============================================================
# PROJECT PATH SETUP
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

SRC_PATH = os.path.join(
    PROJECT_ROOT,
    "src"
)

if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from src.ui import (
    inject_theme_css,
    render_sidebar_brand,
    render_disclaimer_footer,
    section_title,
    loading_spinner,
    render_error_state,
    render_empty_state,
    render_alert,
    mini_stat_html,
    pro_card,
    render_page_hero,
)


# ============================================================
# SKLEARN / XGB IMPORTS (LAZY IMPORT TO SAVE STARTUP TIME)
# ============================================================

def import_ml_deps():

    from sklearn.model_selection import (
        train_test_split,
        StratifiedKFold,
        cross_val_predict
    )

    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
        confusion_matrix
    )

    return (
        train_test_split,
        StratifiedKFold,
        cross_val_predict,
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
        confusion_matrix
    )


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Model Comparison · Loan Approval DSS",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

inject_theme_css()
render_sidebar_brand(
    app_name="Model Comparison",
    subtitle="Base vs Tuned Performance Benchmarks"
)


# ============================================================
# FILE PATHS
# ============================================================

MODELS_PATH = os.path.join(
    PROJECT_ROOT,
    "models"
)


# ============================================================
# SIDEBAR
# ============================================================

# ============================================================
# PAGE HERO
# ============================================================

render_page_hero(
    chip_label="MODEL BENCHMARKING",
    title="Machine Learning Model Comparison",
    subtitle=(
        "Side-by-side benchmark of the 5 core classifiers used for loan "
        "approval prediction, with holdout-set metrics, per-model confusion "
        "matrices, stratified 5-fold cross-validation results, "
        "hyperparameter tuning deltas and visual performance charts."
    ),
)


# ============================================================
# LOAD PERSISTED RESULT CSVs
# ============================================================

eval_path = os.path.join(
    PROJECT_ROOT,
    "evaluation_results.csv"
)

cv_path = os.path.join(
    PROJECT_ROOT,
    "model_results.csv"
)

tuned_eval_path = os.path.join(
    PROJECT_ROOT,
    "tuned_model_results.csv"
)

holdout_df = None
cv_df = None
tuned_eval_df = None

with loading_spinner("Loading evaluation result CSVs …"):
    if os.path.exists(eval_path):
        holdout_df = pd.read_csv(eval_path)

    if os.path.exists(cv_path):
        cv_df = pd.read_csv(cv_path)

    if os.path.exists(tuned_eval_path):
        tuned_eval_df = pd.read_csv(tuned_eval_path)


# ============================================================
# COMPUTE CONFUSION MATRICES ON-DEMAND
# ============================================================

@st.cache_data(show_spinner=False)
def compute_confusion_matrices():
    """
    Reload the saved models + preprocessor, run predictions on the
    canonical 80/20 stratified split, and return a dict with both
    base + tuned confusion matrices + y_test distribution.
    """

    from src.preprocessing import (
        load_data,
        add_features,
        split_features_target
    )

    (
        train_test_split,
        StratifiedKFold,
        cross_val_predict,
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
        confusion_matrix
    ) = import_ml_deps()

    # 1. Reproduce the exact train/test split used for evaluation
    df = load_data()
    df = add_features(df)
    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # 2. Load all persisted models (base + tuned)
    model_filenames = {
        "Logistic Regression": "logistic_regression.pkl",
        "Decision Tree": "decision_tree.pkl",
        "Random Forest": "random_forest.pkl",
        "SVM": "svm.pkl",
        "XGBoost": "xgboost.pkl",
        "Random Forest Tuned": "random_forest_tuned.pkl",
        "SVM Tuned": "svm_tuned.pkl",
        "XGBoost Tuned": "xgboost_tuned.pkl"
    }

    loaded_models = {}
    load_errors = []
    for name, fname in model_filenames.items():
        fpath = Path(MODELS_PATH) / fname
        if fpath.exists():
            try:
                loaded_models[name] = joblib.load(fpath)
            except Exception as e:
                load_errors.append(f"{name} ({fname}): {e}")

    if load_errors:
        st.warning("Some saved models could not be loaded: " + "; ".join(load_errors))

    # 3. Predict + build confusion matrices
    matrices = {}

    for name, model in loaded_models.items():

        y_pred = model.predict(X_test)

        cm = confusion_matrix(y_test, y_pred).tolist()

        matrices[name] = {
            "cm": cm,
            "tn": cm[0][0],
            "fp": cm[0][1],
            "fn": cm[1][0],
            "tp": cm[1][1],
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": (
                roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
                if hasattr(model, "predict_proba")
                else None
            )
        }

    actual_dist = {
        "Actual Rejected": int((y_test == 0).sum()),
        "Actual Approved": int((y_test == 1).sum()),
        "Holdout Size": int(len(y_test))
    }

    return matrices, actual_dist


with loading_spinner(
    "Loading saved models and computing confusion matrices …"
):

    matrices, actual_holdout = compute_confusion_matrices()


# ============================================================
# MODEL CATALOGUE (Top-level selector)
# ============================================================

BASE_MODELS = [
    "Logistic Regression",
    "Decision Tree",
    "Random Forest",
    "SVM",
    "XGBoost"
]

TUNED_MODELS = [
    "Random Forest Tuned",
    "SVM Tuned",
    "XGBoost Tuned"
]

ALL_MODELS = BASE_MODELS + TUNED_MODELS


st.sidebar.markdown("### ⚙️ Filters")

with st.sidebar:

    include_base = st.checkbox(
        "Include Base Models",
        value=True
    )

    include_tuned = st.checkbox(
        "Include Tuned Models",
        value=True
    )

    multi_select = st.multiselect(
        "Specific Models",
        options=ALL_MODELS,
        default=ALL_MODELS
    )

    selected_metric = st.selectbox(
        "Primary Chart Metric",
        options=[
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "ROC-AUC"
        ],
        index=0
    )


active_models = [
    m for m in multi_select
    if (m in BASE_MODELS and include_base) or
       (m in TUNED_MODELS and include_tuned)
]

if not active_models:
    st.warning(
        "No models selected. Use the sidebar filters to enable at "
        "least one base or tuned variant."
    )
    st.stop()


# ============================================================
# 1. TOP-LEVEL KPIs
# ============================================================

section_title("🏆 Best-in-Class Results")

if holdout_df is not None and tuned_eval_df is not None:

    all_holdout = pd.concat(
        [holdout_df, tuned_eval_df],
        ignore_index=True,
        sort=False
    )

else:
    all_holdout = holdout_df if holdout_df is not None else tuned_eval_df

if all_holdout is not None:

    all_holdout = all_holdout[
        all_holdout["Model"].isin(active_models)
    ].copy()

    kpi_c1, kpi_c2, kpi_c3, kpi_c4, kpi_c5 = st.columns(5)

    best_acc_row = all_holdout.loc[all_holdout["Accuracy"].idxmax()]
    best_prec_row = all_holdout.loc[all_holdout["Precision"].idxmax()]
    best_rec_row = all_holdout.loc[all_holdout["Recall"].idxmax()]
    best_f1_row = all_holdout.loc[all_holdout["F1 Score"].idxmax()]
    best_roc_row = all_holdout.loc[all_holdout["ROC-AUC"].idxmax()]

    kpi_c1.metric(
        "Best Accuracy",
        f"{best_acc_row['Accuracy'] * 100:.2f}%",
        best_acc_row["Model"]
    )
    kpi_c2.metric(
        "Best Precision",
        f"{best_prec_row['Precision'] * 100:.2f}%",
        best_prec_row["Model"]
    )
    kpi_c3.metric(
        "Best Recall",
        f"{best_rec_row['Recall'] * 100:.2f}%",
        best_rec_row["Model"]
    )
    kpi_c4.metric(
        "Best F1 Score",
        f"{best_f1_row['F1 Score'] * 100:.2f}%",
        best_f1_row["Model"]
    )
    kpi_c5.metric(
        "Best ROC-AUC",
        f"{best_roc_row['ROC-AUC'] * 100:.2f}%",
        best_roc_row["Model"]
    )

    st.markdown("")


st.markdown("---")

# ============================================================
# 2. CONSOLIDATED PERFORMANCE TABLE
#    (Accuracy, Precision, Recall, F1, ROC-AUC, CV Mean, CV Std)
# ============================================================

section_title("📋 Consolidated Metrics (Holdout + CV)")

def build_consolidated_table():

    frames = []
    if holdout_df is not None:
        base = holdout_df.copy()
        base["Variant"] = "Base"
        frames.append(base)
    if tuned_eval_df is not None:
        tuned = tuned_eval_df.copy()
        tuned["Variant"] = "Tuned"
        frames.append(tuned)

    if not frames:
        return None

    perf = pd.concat(frames, ignore_index=True, sort=False)

    if cv_df is not None:
        perf = perf.merge(
            cv_df,
            on="Model",
            how="left"
        )
    else:
        perf["CV Mean Accuracy"] = np.nan
        perf["CV Std"] = np.nan

    perf = perf[
        perf["Model"].isin(active_models)
    ].copy()

    col_order = [
        "Model", "Variant",
        "Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC",
        "CV Mean Accuracy", "CV Std"
    ]
    col_order = [c for c in col_order if c in perf.columns]
    perf = perf[col_order]

    return perf


consolidated = build_consolidated_table()

if consolidated is not None:

    numeric_cols = [
        c for c in consolidated.columns
        if c not in ("Model", "Variant")
    ]

    max_highlight_cols = [
        "Accuracy", "Precision", "Recall",
        "F1 Score", "ROC-AUC", "CV Mean Accuracy"
    ]
    max_highlight_cols = [
        c for c in max_highlight_cols if c in consolidated.columns
    ]

    styled = consolidated.style.format(
        subset=numeric_cols,
        formatter=lambda x: (
            f"{x:.4f}" if pd.notna(x) else "—"
        )
    ).highlight_max(
        subset=max_highlight_cols,
        color="#d4edda"
    )

    if "CV Std" in consolidated.columns:
        styled = styled.highlight_min(
            subset=["CV Std"],
            color="#d4edda"
        )

    # Variant badges
    def _variant_badge(v):
        if v == "Base":
            return '<span class="base-badge">BASE</span>'
        if v == "Tuned":
            return '<span class="tuned-badge">TUNED</span>'
        return str(v)

    if "Variant" in consolidated.columns:
        styled = styled.format(
            subset=["Variant"],
            formatter=_variant_badge
        )

    st.dataframe(
        styled,
        width='stretch',
        hide_index=True,
        height=32 + 35 * len(consolidated)
    )

    with st.expander(
        "📖 Metric Glossary",
        expanded=False
    ):
        st.markdown(
            """
            | Metric | Formula / Meaning |
            |--------|-------------------|
            | **Accuracy** | `(TP + TN) / All` — Overall correctness. |
            | **Precision** | `TP / (TP + FP)` — Of predicted Approved, how many were real? (Minimises *false approvals*). |
            | **Recall** (Sensitivity) | `TP / (TP + FN)` — Of real Approved applications, how many were caught? (Minimises *missed approvals*). |
            | **F1 Score** | `2·P·R / (P + R)` — Harmonic mean of Precision & Recall. |
            | **ROC-AUC** | Area under the Receiver-Operating-Characteristic curve; ranking-quality metric. |
            | **CV Mean Accuracy** | Mean accuracy over **5-fold stratified CV** on the training set. |
            | **CV Std** | Standard deviation of those 5 fold scores — lower = more stable. |
            """
        )

else:

    st.info(
        "No evaluation CSVs available (evaluation_results.csv, "
        "tuned_model_results.csv). Confusion matrices and charts "
        "are still available from the loaded models below."
    )


st.markdown("---")

# ============================================================
# 3. PERFORMANCE COMPARISON CHARTS
# ============================================================

section_title("📈 Performance Comparison Charts")

chart_c1, chart_c2 = st.columns(2)

with chart_c1:

    st.markdown(
        f"**Selected Metric: {selected_metric}**"
    )

    if consolidated is not None and selected_metric in consolidated.columns:

        bar_df = (
            consolidated[["Model", selected_metric]]
            .dropna()
            .set_index("Model")
            .sort_values(selected_metric, ascending=True)
        )

        st.bar_chart(
            bar_df,
            width='stretch',
            horizontal=True,
            height=360
        )

    else:

        st.info(
            f"{selected_metric} data is not available for the current selection."
        )

with chart_c2:

    st.markdown(
        "**All 5 Metrics Comparison (Grouped)**"
    )

    if consolidated is not None:

        metric_cols = [
            "Accuracy", "Precision", "Recall",
            "F1 Score", "ROC-AUC"
        ]
        metric_cols = [
            c for c in metric_cols if c in consolidated.columns
        ]

        grouped = (
            consolidated[["Model"] + metric_cols]
            .dropna()
            .set_index("Model")
        )

        st.line_chart(
            grouped,
            width='stretch',
            height=360
        )

    else:

        st.info(
            "Holdout metrics are required for the grouped chart."
        )

st.markdown("")

radar_c1, radar_c2 = st.columns(2)

with radar_c1:

    st.markdown(
        "**F1 Score vs ROC-AUC Scatter**"
    )

    if consolidated is not None and \
       "F1 Score" in consolidated.columns and \
       "ROC-AUC" in consolidated.columns:

        scatter_df = consolidated[
            ["Model", "F1 Score", "ROC-AUC"]
        ].dropna().copy()
        scatter_df["F1 Score (%)"] = scatter_df["F1 Score"] * 100
        scatter_df["ROC-AUC (%)"] = scatter_df["ROC-AUC"] * 100

        st.scatter_chart(
            scatter_df.set_index("Model")[
                ["F1 Score (%)", "ROC-AUC (%)"]
            ],
            width='stretch',
            height=340,
            size=60
        )

    else:

        st.info("F1 / ROC-AUC data unavailable.")

with radar_c2:

    st.markdown(
        "**CV Mean Accuracy (± CV Std)**"
    )

    if consolidated is not None and \
       "CV Mean Accuracy" in consolidated.columns:

        cv_chart_df = (
            consolidated[["Model", "CV Mean Accuracy", "CV Std"]]
            .dropna()
            .set_index("Model")
            .sort_values("CV Mean Accuracy", ascending=True)
        )

        if not cv_chart_df.empty:

            st.bar_chart(
                cv_chart_df[["CV Mean Accuracy"]],
                width='stretch',
                horizontal=True,
                height=340
            )

            with st.expander("CV Std (stability) values"):
                st.dataframe(
                    cv_chart_df.reset_index()[[
                        "Model", "CV Mean Accuracy", "CV Std"
                    ]].style.format({
                        "CV Mean Accuracy": "{:.4f}",
                        "CV Std": "{:.4f}"
                    }),
                    width='stretch',
                    hide_index=True
                )

        else:

            st.info("Cross-validation not available for selection.")

    else:

        st.info("Cross-validation data unavailable.")


st.markdown("#### 🌳 Decision Tree Graph")
st.caption(
    "The first four levels of the trained Decision Tree, shown alongside the "
    "model comparison charts."
)

try:
    from sklearn.tree import plot_tree

    tree_pipeline = joblib.load(Path(MODELS_PATH) / "decision_tree.pkl")
    tree_steps = getattr(tree_pipeline, "named_steps", {})
    tree_estimator = tree_steps.get("model", tree_pipeline)
    tree_preprocessor = tree_steps.get("preprocessor")
    tree_features = None
    if tree_preprocessor is not None and hasattr(tree_preprocessor, "get_feature_names_out"):
        try:
            tree_features = [
                name.split("__", 1)[-1]
                for name in tree_preprocessor.get_feature_names_out()
            ]
        except Exception:
            tree_features = None

    tree_fig, tree_ax = plt.subplots(figsize=(22, 9))
    plot_tree(
        tree_estimator,
        feature_names=tree_features,
        class_names=["Rejected", "Approved"],
        filled=True,
        rounded=True,
        max_depth=3,
        fontsize=8,
        ax=tree_ax
    )
    tree_ax.set_title("Decision Tree — Loan Approval")
    tree_fig.tight_layout()
    st.pyplot(tree_fig)
    plt.close(tree_fig)
except Exception as e:
    st.warning(f"Decision Tree graph could not be loaded: {e}")


st.markdown("---")

# ============================================================
# 4. CONFUSION MATRICES
# ============================================================

section_title("🧩 Confusion Matrices (Holdout Set)")

st.caption(
    f"Holdout: **{actual_holdout.get('Actual Rejected','?')} Rejected · "
    f"{actual_holdout.get('Actual Approved','?')} Approved · "
    f"{actual_holdout.get('Holdout Size','?')} samples** "
    "(stratified 20% split, random_state=42)."
)

st.markdown("")

active_cm_models = [m for m in active_models if m in matrices]

if not active_cm_models:

    st.info(
        "Confusion matrices are not yet available for the selected models."
    )

else:

    for row_start in range(0, len(active_cm_models), 2):

        row_cols = st.columns(2)

        for offset in range(2):

            idx = row_start + offset

            if idx < len(active_cm_models):

                model = active_cm_models[idx]
                m = matrices[model]

                with row_cols[offset]:
                    tag_label = "TUNED" if "Tuned" in model else "BASE"
                    with st.container(border=True):
                        st.markdown(f"#### 🧠 {model}")
                        st.caption(tag_label)
                        confusion_df = pd.DataFrame(
                            [
                                [m["tn"], m["fp"]],
                                [m["fn"], m["tp"]],
                            ],
                            index=["Actual Rejected", "Actual Approved"],
                            columns=["Predicted Rejected", "Predicted Approved"]
                        )
                        st.dataframe(confusion_df, width="stretch")

                        metric_rows = [
                            ("Accuracy", m["accuracy"]),
                            ("Precision", m["precision"]),
                            ("Recall", m["recall"]),
                            ("F1", m["f1"]),
                            ("ROC-AUC", m["roc_auc"]),
                            ("Error Rate", 1 - m["accuracy"]),
                        ]
                        for metric_start in (0, 3):
                            metric_cols = st.columns(3)
                            for col, (label, value) in zip(
                                metric_cols, metric_rows[metric_start:metric_start + 3]
                            ):
                                with col:
                                    display_value = "—" if value is None else f"{value * 100:.1f}%"
                                    st.metric(label, display_value)

        st.markdown("")


st.markdown("---")

# ============================================================
# 5. CROSS-VALIDATION RESULTS
# ============================================================

section_title("🔁 Cross-Validation Results")

if cv_df is not None:

    cv_active = cv_df[
        cv_df["Model"].isin(active_models)
    ].copy()

    if not cv_active.empty:

        cv_active = cv_active.sort_values(
            "CV Mean Accuracy",
            ascending=False
        )

        cv_k1, cv_k2, cv_k3 = st.columns(3)

        cv_k1.metric(
            "Highest CV Mean Accuracy",
            f"{cv_active['CV Mean Accuracy'].max()*100:.2f}%",
            cv_active["CV Mean Accuracy"].idxmax()
            if isinstance(cv_active["CV Mean Accuracy"].idxmax(), str)
            else cv_active.loc[
                cv_active["CV Mean Accuracy"].idxmax(),
                "Model"
            ]
        )

        cv_k2.metric(
            "Most Stable (Lowest CV Std)",
            f"{cv_active['CV Std'].min()*100:.2f}%",
            cv_active.loc[
                cv_active["CV Std"].idxmin(),
                "Model"
            ]
        )

        cv_k3.metric(
            "Avg CV Mean (Selection)",
            f"{cv_active['CV Mean Accuracy'].mean()*100:.2f}%"
        )

        st.markdown("")

        cv_c1, cv_c2 = st.columns([1, 1.3])

        with cv_c1:

            st.subheader("CV Results Table")

            st.dataframe(
                cv_active.style.format({
                    "CV Mean Accuracy": "{:.4f}",
                    "CV Std": "{:.4f}"
                }).bar(
                    subset=["CV Mean Accuracy"],
                    color="#93c5fd"
                ),
                width='stretch',
                hide_index=True,
                height=300
            )

        with cv_c2:

            st.subheader("CV Mean Accuracy ± Stability Band")

            chart_data = (
                cv_active[["Model", "CV Mean Accuracy", "CV Std"]]
                .set_index("Model")
                .sort_values("CV Mean Accuracy", ascending=True)
            )

            st.bar_chart(
                chart_data["CV Mean Accuracy"],
                width='stretch',
                horizontal=True,
                height=300
            )

            st.caption(
                "Each bar = CV Mean Accuracy. Lower CV Std means the model "
                "is more consistent across folds — inspect the table for exact values."
            )

        with st.expander(
            "🔬 About the Cross-Validation Setup",
            expanded=False
        ):

            st.markdown(
                """
                - **Splitter:** `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
                - **Scoring:** Accuracy
                - **Training data:** 80% of the dataset (stratified to preserve the
                  68.7% / 31.3% Approved/Rejected class ratio)
                - **Purpose:** Estimate generalisation performance *without* tuning,
                  and to rank model families before hyperparameter search.
                """
            )

    else:

        st.info(
            "CV results only include base models — check **Include Base Models** "
            "in the sidebar to see them."
        )

else:

    st.info(
        "Cross-validation file `model_results.csv` was not found in the "
        "project root. Re-run `src/train_models.py` to regenerate it."
    )


st.markdown("---")

# ============================================================
# 6. HYPERPARAMETER TUNING RESULTS
# ============================================================

section_title("⚙️ Hyperparameter Tuning Results")

# Best-tuned params (hard-coded from src/tune_models.py search space +
# best-known values; these are the canonical values the saved
# *_tuned.pkl pipelines were selected with by RandomizedSearchCV).
# If the project later saves tuning artifacts, we can load them instead.

TUNING_INFO = {
    "Random Forest Tuned": {
        "base_model": "Random Forest",
        "search_method": "RandomizedSearchCV",
        "n_iter": 15,
        "cv_folds": 5,
        "scoring": "ROC-AUC",
        "search_space": [
            "n_estimators ∈ {100, 200, 300}",
            "max_depth ∈ {None, 5, 8, 10, 15}",
            "min_samples_split ∈ {2, 5, 10}",
            "min_samples_leaf ∈ {1, 2, 4}",
            "max_features ∈ {sqrt, log2}"
        ]
    },
    "SVM Tuned": {
        "base_model": "SVM",
        "search_method": "RandomizedSearchCV",
        "n_iter": 15,
        "cv_folds": 5,
        "scoring": "ROC-AUC",
        "search_space": [
            "C ∈ {0.1, 1, 10, 100}",
            "gamma ∈ {scale, auto, 0.01, 0.1}",
            "kernel ∈ {rbf, linear}"
        ]
    },
    "XGBoost Tuned": {
        "base_model": "XGBoost",
        "search_method": "RandomizedSearchCV",
        "n_iter": 15,
        "cv_folds": 5,
        "scoring": "ROC-AUC",
        "search_space": [
            "n_estimators ∈ {100, 200, 300}",
            "max_depth ∈ {2, 3, 4, 5, 6}",
            "learning_rate ∈ {0.01, 0.05, 0.1, 0.2}",
            "subsample ∈ {0.7, 0.8, 1.0}",
            "colsample_bytree ∈ {0.7, 0.8, 1.0}"
        ]
    }
}

if holdout_df is not None and tuned_eval_df is not None:

    # Build base vs tuned side-by-side with deltas
    tuning_compare_rows = []

    for tuned_model, info in TUNING_INFO.items():

        if tuned_model not in active_models:
            continue
        if info["base_model"] not in holdout_df["Model"].values:
            continue
        if tuned_model not in tuned_eval_df["Model"].values:
            continue

        base_row = holdout_df[
            holdout_df["Model"] == info["base_model"]
        ].iloc[0]

        tuned_row = tuned_eval_df[
            tuned_eval_df["Model"] == tuned_model
        ].iloc[0]

        for metric in [
            "Accuracy", "Precision", "Recall",
            "F1 Score", "ROC-AUC"
        ]:
            tuning_compare_rows.append({
                "Tuned Model": tuned_model,
                "Base Model": info["base_model"],
                "Metric": metric,
                "Base": base_row[metric],
                "Tuned": tuned_row[metric],
                "Δ (abs)": tuned_row[metric] - base_row[metric],
                "Δ (%)": (
                    (tuned_row[metric] - base_row[metric]) * 100
                )
            })

    if tuning_compare_rows:

        tuning_df = pd.DataFrame(tuning_compare_rows)

        # ---- Summary KPI: best tuning lift ----

        tuning_kpi_c1, tuning_kpi_c2, tuning_kpi_c3 = st.columns(3)

        max_delta_row = tuning_df.loc[tuning_df["Δ (abs)"].idxmax()]

        tuning_kpi_c1.metric(
            "Largest Tuning Gain",
            f"+{max_delta_row['Δ (%)']:.2f} pp",
            f"{max_delta_row['Tuned Model']} · {max_delta_row['Metric']}"
        )

        avg_delta = tuning_df["Δ (%)"].mean()
        tuning_kpi_c2.metric(
            "Avg Tuning Δ across all metrics",
            f"{avg_delta:+.2f} pp"
        )

        # Count how many metric deltas are non-negative
        improved_count = int((tuning_df["Δ (abs)"] >= 0).sum())
        total_count = len(tuning_df)
        tuning_kpi_c3.metric(
            "Metrics Improved / Maintained",
            f"{improved_count}/{total_count}",
            (
                "All improved!"
                if improved_count == total_count
                else f"{(improved_count/total_count)*100:.0f}%"
            )
        )

        st.markdown("")

        t_c1, t_c2 = st.columns(2)

        with t_c1:

            st.subheader("Base vs Tuned — Holdout Metrics")

            def _color_delta(val):
                if val > 0.00001:
                    return 'color:#16a34a;font-weight:700;'
                if val < -0.00001:
                    return 'color:#dc2626;font-weight:700;'
                return 'color:#666;'

            styled_tuning = tuning_df.style.format({
                "Base": "{:.4f}",
                "Tuned": "{:.4f}",
                "Δ (abs)": "{:+.4f}",
                "Δ (%)": "{:+.2f} pp"
            }).map(
                _color_delta,
                subset=["Δ (abs)", "Δ (%)"]
            )

            st.dataframe(
                styled_tuning,
                width='stretch',
                hide_index=True,
                height=30 + 35 * len(tuning_df)
            )

        with t_c2:

            st.subheader("Tuning Δ (%) by Model × Metric")

            pivot = tuning_df.pivot(
                index="Tuned Model",
                columns="Metric",
                values="Δ (%)"
            )

            def _cell_bg(v):
                if v > 0:
                    intensity = min(abs(v) * 80, 230)
                    r, g, b = 220 - int(intensity * 0.3), 252, 231
                    return f"background-color:rgb({r},{g},{b});"
                if v < 0:
                    intensity = min(abs(v) * 80, 230)
                    r, g, b = 254, 226 - int(intensity * 0.3), 226
                    return f"background-color:rgb({r},{g},{b});"
                return ""

            st.dataframe(
                pivot.style.format("{:+.2f} pp").map(_cell_bg),
                width='stretch',
                height=180
            )

        st.markdown("")

        # ---- Tuned model detail cards ----

        st.subheader("Tuning Search Configurations")

        active_tuned = [
            m for m in TUNING_INFO if m in active_models
        ]

        for i in range(0, len(active_tuned), 2):

            cols = st.columns(2)

            for j in range(2):

                idx = i + j
                if idx < len(active_tuned):

                    tm = active_tuned[idx]
                    info = TUNING_INFO[tm]

                    with cols[j]:

                        base_perf = (
                            holdout_df[
                                holdout_df["Model"] == info["base_model"]
                            ]["Accuracy"].values[0] * 100
                            if info["base_model"] in holdout_df["Model"].values
                            else None
                        )
                        tuned_perf = (
                            tuned_eval_df[
                                tuned_eval_df["Model"] == tm
                            ]["Accuracy"].values[0] * 100
                            if tm in tuned_eval_df["Model"].values
                            else None
                        )
                        delta_acc = (
                            f"{tuned_perf - base_perf:+.2f} pp"
                            if base_perf and tuned_perf
                            else "—"
                        )

                        with st.container(border=True):
                            st.markdown(f"#### {tm} · Tuned")
                            st.write(
                                f"Baseline: **{info['base_model']}** · "
                                f"Strategy: **{info['search_method']}** · "
                                f"Scoring: **{info['scoring']}**"
                            )
                            tuned_cols = st.columns(3)
                            tuned_cols[0].metric(
                                "Base Accuracy",
                                "—" if base_perf is None else f"{base_perf:.2f}%"
                            )
                            tuned_cols[1].metric(
                                "Tuned Accuracy",
                                "—" if tuned_perf is None else f"{tuned_perf:.2f}%"
                            )
                            tuned_cols[2].metric("Accuracy Change", delta_acc)
                            st.caption(
                                f"Search space: {info['n_iter']} iterations · "
                                f"{info['cv_folds']}-fold cross-validation"
                            )
                            st.write(info["search_space"])

            st.markdown("")

    else:

        st.info(
            "No tuning comparison rows available. Ensure at least one "
            "tuned variant and its base model are selected in the sidebar."
        )

else:

    st.info(
        "Hyperparameter tuning results require both `evaluation_results.csv` "
        "(base models) and `tuned_model_results.csv` (tuned variants)."
    )


render_disclaimer_footer()
