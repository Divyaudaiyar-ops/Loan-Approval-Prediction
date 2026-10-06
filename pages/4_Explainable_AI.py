import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sqlite3
import os
import sys

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


from src.preprocessing import add_features

from src.ui import (
    inject_theme_css,
    render_sidebar_brand,
    section_title,
    loading_spinner,
    render_error_state,
    render_empty_state,
    render_alert,
    risk_badge_html,
    mini_stat_html,
    pro_card,
    render_page_hero,
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Explainable AI · Loan Approval DSS",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

inject_theme_css()
render_sidebar_brand(
    app_name="Explainable AI",
    subtitle="Global SHAP Feature Importance"
)


# ============================================================
# FILE PATHS
# ============================================================

SHAP_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "shap_results"
)

IMPORTANCE_CSV = os.path.join(
    SHAP_DIR,
    "shap_feature_importance.csv"
)
IMPORTANCE_PNG = os.path.join(
    SHAP_DIR,
    "shap_feature_importance.png"
)
SUMMARY_PNG = os.path.join(
    SHAP_DIR,
    "shap_summary.png"
)

MODELS_DIR = os.path.join(
    PROJECT_ROOT,
    "models"
)

DB_PATH = os.path.join(
    PROJECT_ROOT,
    "loan_predictions.db"
)


# ============================================================
# SIDEBAR
# ============================================================

# ============================================================
# PAGE HERO
# ============================================================

render_page_hero(
    chip_label="SHAP EXPLAINABILITY",
    title="Explainable Artificial Intelligence (SHAP)",
    subtitle=(
        "Global SHAP charts summarize which features influenced the trained "
        "models across the evaluation data. You can also inspect an individual "
        "classifier's output; final recommendations and affordability review "
        "are calculated on the main Prediction page."
    ),
)


# ============================================================
# HELPERS
# ============================================================

FEATURE_LABELS = {
    "num__Credit_History": "💳 Credit History",
    "num__Total_Income": "👥 Total Household Income",
    "num__Loan_Amount_Term": "⏱️ Loan Term (months)",
    "num__Loan_Amount": "💰 Requested Loan Amount",
    "num__Dependents_Standardized": "👨‍👩‍👧 Dependents (scaled)",
    "cat__Education_Not Graduate": "📚 Education = Not Graduate",
    "cat__Education_Graduate": "🎓 Education = Graduate",
    "cat__Self_Employed_No": "💼 Self-Employed = No",
    "cat__Self_Employed_Yes": "🧑‍💼 Self-Employed = Yes",
    "cat__Property_Area_Rural": "🌾 Property Area = Rural",
    "cat__Property_Area_Semiurban": "🏘️ Property Area = Semiurban",
    "cat__Property_Area_Urban": "🏙️ Property Area = Urban",
    "ord__Dependents_Ordinal": "👨‍👩‍👧 Dependents (ordinal)",
    "remainder__Dependents": "👨‍👩‍👧 Dependents (remainder)"
}


def humanise_feature(raw_name: str) -> str:
    if raw_name in FEATURE_LABELS:
        return FEATURE_LABELS[raw_name]
    nice = (
        raw_name
        .replace("num__", "")
        .replace("cat__", "")
        .replace("ord__", "")
        .replace("remainder__", "")
        .replace("_", " ")
        .title()
    )
    return f"🔹 {nice}"


def load_prediction_history():
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    conn = sqlite3.connect(DB_PATH)
    try:
        hist = pd.read_sql_query(
            "SELECT * FROM predictions ORDER BY id DESC",
            conn
        )
    finally:
        conn.close()
    return hist


def history_row_to_input_df(row: pd.Series) -> pd.DataFrame:
    """Best-effort reconstruct a DataFrame suitable for add_features
    and subsequent prediction from a prediction-history row.
    """

    return pd.DataFrame({
        "Dependents": [0],    # are sourced from the history row below
        "Education": [row["education"] if pd.notna(row.get("education")) else "Graduate"],
        "Self_Employed": ["No"],
        "ApplicantIncome": [float(row.get("applicant_income", 0) or 0)],
        "CoapplicantIncome": [float(row.get("coapplicant_income", 0) or 0)],
        "LoanAmount": [float(row.get("loan_amount", 0) or 0)],
        "Loan_Amount_Term": [360.0],
        "Credit_History": [float(row.get("credit_history", 1.0) or 0)],
        "Property_Area": [row["property_area"] if pd.notna(row.get("property_area")) else "Semiurban"]
    })


# ============================================================
# GLOBAL SHAP FEATURE IMPORTANCE
# ============================================================

section_title("🌐 SHAP Global Feature Importance")

shap_importance_df = None

with loading_spinner("Loading SHAP global importance artifacts …"):
    if os.path.exists(IMPORTANCE_CSV):
        shap_importance_df = pd.read_csv(IMPORTANCE_CSV)

if shap_importance_df is not None:

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    total_importance = shap_importance_df["SHAP Importance"].sum()

    top1_row = shap_importance_df.iloc[0]
    top1_share = (top1_row["SHAP Importance"] / total_importance * 100)

    top3_share = (
        shap_importance_df.head(3)["SHAP Importance"].sum()
        / total_importance * 100
    )

    kpi1.metric(
        "Transformed Features Explained",
        len(shap_importance_df)
    )
    kpi2.metric(
        "Most Influential Feature",
        humanise_feature(top1_row["Feature"]),
        f"{top1_share:.1f}% of total SHAP"
    )
    kpi3.metric(
        "Top-3 Features Share",
        f"{top3_share:.1f}%"
    )
    kpi4.metric(
        "Mean |SHAP| Value",
        f"{shap_importance_df['SHAP Importance'].mean():.4f}"
    )

    st.markdown("")

    gi_c1, gi_c2 = st.columns([1.2, 1])

    with gi_c1:

        st.subheader("Ranked SHAP Importance (Top 15)")

        top15 = shap_importance_df.head(15).copy()
        top15["Feature Label"] = top15["Feature"].map(humanise_feature)

        chart_df = (
            top15[["Feature Label", "SHAP Importance"]]
            .set_index("Feature Label")
            .sort_values("SHAP Importance", ascending=True)
        )

        st.bar_chart(
            chart_df,
            width='stretch',
            horizontal=True,
            height=520
        )

    with gi_c2:

        st.subheader("SHAP Importance Table")

        display_table = shap_importance_df.copy()
        display_table.insert(0, "Rank", np.arange(1, len(display_table) + 1))
        display_table.insert(1, "Feature Label", display_table["Feature"].map(humanise_feature))
        display_table["Share (%)"] = (
            display_table["SHAP Importance"] / total_importance * 100
        ).round(2)
        display_table["Cumulative (%)"] = (
            display_table["Share (%)"].cumsum()
        ).round(2)

        st.dataframe(
            display_table[[
                "Rank", "Feature Label", "Feature",
                "SHAP Importance", "Share (%)", "Cumulative (%)"
            ]].style.format({
                "SHAP Importance": "{:.4f}",
                "Share (%)": "{:.2f} %",
                "Cumulative (%)": "{:.2f} %"
            }).bar(subset=["SHAP Importance"], color="#93c5fd"),
            width='stretch',
            hide_index=True,
            height=500
        )

else:

    render_alert(
        "<code>shap_feature_importance.csv</code> not found in <code>data/shap_results/</code>. "
        "Re-run <code>src/explain.py</code> to regenerate global SHAP artifacts.",
        kind="warning"
    )

st.markdown("---")

# ============================================================
# SHAP SUMMARY PLOT
# ============================================================

section_title("🧠 SHAP Summary Plot (Global Feature Impact)")

sum_c1, sum_c2 = st.columns([1.3, 1])

with sum_c1:

    if os.path.exists(SUMMARY_PNG):
        st.image(
            SUMMARY_PNG,
            caption="Each dot = one test sample. Color = feature value (high/low). "
                    "Horizontal position = direction & magnitude of that feature's "
                    "SHAP contribution towards the Approved class.",
            width='stretch'
        )
    else:
        st.info("Summary plot not available.")

with sum_c2:

    if os.path.exists(IMPORTANCE_PNG):
        st.image(
            IMPORTANCE_PNG,
            caption="Aggregate mean(|SHAP|) ranking plot.",
            width='stretch'
        )
    else:
        st.info("Bar plot not available.")

st.markdown("---")

# ============================================================
# TOP INFLUENCING FEATURES
# ============================================================

section_title("⭐ Top Influencing Features")

if shap_importance_df is not None:

    n_top = st.slider(
        "How many top features to detail?",
        min_value=3, max_value=10, value=5
    )

    top_features = shap_importance_df.head(n_top).reset_index(drop=True)

    for row_i in range(0, len(top_features), 2):

        row_cols = st.columns(2)

        for col_j in range(2):

            idx = row_i + col_j
            if idx >= len(top_features):
                continue

            feat = top_features.iloc[idx]
            rank = idx + 1
            label = humanise_feature(feat["Feature"])
            shap_val = feat["SHAP Importance"]
            share = shap_val / total_importance * 100

            # Produce an "interpretation" snippet
            raw = feat["Feature"]
            if "Credit_History" in raw:
                narrative = (
                    "Having a history of repaying past debts is the single "
                    "strongest driver of approval probability."
                )
            elif "Total_Income" in raw or "ApplicantIncome" in raw:
                narrative = (
                    "Higher applicant / household incomes push approval "
                    "probability upwards, as repayment capacity grows."
                )
            elif "Loan_to_Income" in raw or "LoanAmount" in raw:
                narrative = (
                    "Larger loans (or a higher loan-to-income ratio) "
                    "typically push approval probability DOWN."
                )
            elif "Education" in raw:
                narrative = (
                    "Graduate status correlates with higher repayment "
                    "capacity and higher approval odds."
                )
            elif "Property_Area" in raw:
                narrative = (
                    "Property location captures collateral market risk "
                    "and locality-level approval trends."
                )
            else:
                narrative = (
                    "This feature contributes consistently to the model's "
                    "final decision across the holdout set."
                )

            with row_cols[col_j]:

                st.markdown(
                    f"""
                    <div class="feature-card" style="border:1px solid #cbd5e1;">
                        <div class="feature-headline">
                            <span>
                                <span class="rank-badge">{rank}</span>
                                <span class="feature-name">{label}</span>
                            </span>
                            <span class="contrib-value pos-text">+{shap_val:.4f} · {share:.1f}%</span>
                        </div>
                        <div class="feature-value" style="margin-bottom:8px;">
                            Raw column name: <code>{raw}</code>
                        </div>
                        <div style="font-size:13.5px;color:#374151;line-height:1.5;">
                            💡 {narrative}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

else:

    st.info(
        "Top influencing features are computed from the global SHAP importance CSV."
    )

st.markdown("---")

# ============================================================
# INDIVIDUAL PREDICTION EXPLANATION
# ============================================================

section_title("🧩 Individual Prediction Explanation")

st.markdown(
    """
    Choose a prediction to explain: either pick a **previously logged**
    run from the Prediction History database, or build a **custom
    applicant profile** using the form. Contributions are approximated
    using the **global SHAP importance weights** scaled against the
    direction each feature pushes the final log-odds/probability.
    """
)

with loading_spinner("Loading prediction history from SQLite DB …"):
    history_df = load_prediction_history()

mode = st.radio(
    "Prediction source",
    [
        "📜 Pick from Prediction History",
        "✍️ Build a custom applicant profile"
    ],
    horizontal=True
)

selected_row = None
custom_input = None
prediction_label = None
probability_value = None

if mode == "📜 Pick from Prediction History":

    if history_df.empty:

        st.info(
            "No prediction history records yet. Run at least one prediction "
            "from the 🔮 Prediction page, then return here, or switch to "
            "**Build a custom applicant profile** below."
        )

    else:

        hist_display = history_df[[
            "id", "timestamp", "model", "prediction",
            "probability", "applicant_income", "coapplicant_income",
            "loan_amount", "credit_history", "education", "property_area"
        ]].copy()
        hist_display["probability"] = (
            hist_display["probability"] * 100
        ).round(2).astype(str) + "%"

        st.caption(
            f"{len(history_df):,} past predictions found. "
            "Click a row below to select it for explanation."
        )

        event = st.dataframe(
            hist_display,
            width='stretch',
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            height=260
        )

        sel_rows = event.selection["rows"]

        if sel_rows:

            idx = int(sel_rows[0])
            selected_row = history_df.iloc[idx]

            st.success(
                f"Selected prediction #{selected_row['id']} "
                f"({selected_row['timestamp']}) — "
                f"Model: **{selected_row['model']}**, "
                f"Outcome: **{selected_row['prediction']}**, "
                f"Probability: **{selected_row['probability']*100:.2f}%**"
            )

            prediction_label = selected_row["prediction"]
            probability_value = float(selected_row["probability"])

else:

    # Custom profile builder
    with st.form("custom_profile_form"):

        st.markdown(
            "**Applicant Profile**"
        )

        c1, c2, c3 = st.columns(3)

        with c1:
            cp_credit = st.selectbox(
                "Credit History", [1.0, 0.0],
                format_func=lambda v: "✅ Good (1.0)" if v == 1.0 else "❌ Poor (0.0)"
            )
            cp_education = st.selectbox(
                "Education", ["Graduate", "Not Graduate"]
            )

        with c2:
            cp_applicant_inc = st.number_input(
                "Applicant Income", min_value=0.0, value=5000.0, step=100.0
            )
            cp_coapplicant_inc = st.number_input(
                "Coapplicant Income", min_value=0.0, value=0.0, step=100.0
            )

        with c3:
            cp_loan_amount = st.number_input(
                "Loan Amount", min_value=0.0, value=150.0, step=10.0
            )
            cp_loan_term = st.selectbox(
                "Loan Term (months)",
                [360, 300, 240, 180, 120, 84, 60, 36]
            )
            cp_property_area = st.selectbox(
                "Property Area", ["Semiurban", "Urban", "Rural"]
            )

        cp_dependents = st.selectbox("Dependents", [0, 1, 2, 3])
        cp_self_employed = st.selectbox("Self Employed", ["No", "Yes"])

        cp_model = st.selectbox(
            "Model (to run the prediction)",
            list({
                "Random Forest Tuned": None,
                "XGBoost Tuned": None,
                "SVM Tuned": None,
                "Logistic Regression": None,
                "Random Forest": None,
                "XGBoost": None,
                "SVM": None,
                "Decision Tree": None
            }.keys())
        )

        run_custom = st.form_submit_button(
            "Run Prediction & Explain",
            width='stretch'
        )

    if run_custom:

        custom_input = pd.DataFrame({
            "Dependents": [cp_dependents],
            "Education": [cp_education],
            "Self_Employed": [cp_self_employed],
            "ApplicantIncome": [cp_applicant_inc],
            "CoapplicantIncome": [cp_coapplicant_inc],
            "LoanAmount": [cp_loan_amount],
            "Loan_Amount_Term": [cp_loan_term],
            "Credit_History": [cp_credit],
            "Property_Area": [cp_property_area]
        })

        model_path = Path(MODELS_DIR) / (
            cp_model.lower().replace(" ", "_") + ".pkl"
        )

        if model_path.exists():

            with loading_spinner(f"Running {cp_model} prediction & computing probabilities …"):
                try:
                    model = joblib.load(model_path)
                except Exception as e:
                    render_error_state(
                        f"Failed to load model: {str(e)}",
                        hint="Check that model dependencies (e.g. xgboost) are installed."
                    )
                    st.stop()
                processed = add_features(custom_input.copy())

                pred = int(model.predict(processed)[0])
                prob = float(model.predict_proba(processed)[0][1])

            prediction_label = "Positive class" if pred == 1 else "Negative class"
            probability_value = prob

            st.success(
                f"Individual model result from **{cp_model}** → "
                f"**{prediction_label}** with probability "
                f"**{probability_value * 100:.2f}%**."
            )

        else:

            render_error_state(
                f"Model pickle not found at `{model_path}`.",
                hint="Ensure the model has been trained and saved to the models/ folder."
            )


# ============================================================
# CUSTOM / HISTORICAL MODEL OUTPUT
# ============================================================

if (selected_row is not None or custom_input is not None) and \
   prediction_label is not None and probability_value is not None:
    section_title("🔎 Selected Model Result")
    st.metric("Model", model_used, help="This is one model's output; the main Prediction page combines five models and applies the affordability policy.")
    st.metric("Individual model approval probability", f"{probability_value * 100:.2f}%")
    st.info(
        f"This individual model labels the application **{prediction_label}**. "
        "The final dashboard recommendation also considers the five-model average "
        "and payment-to-income DTI, so the final Approved/Rejected status may "
        "differ from this one model's result."
    )
    st.caption(
        "The chart above shows global SHAP importance from the trained models. "
        "This page does not calculate local SHAP values for the selected applicant."
    )

st.markdown("---")
st.info(
    "💡 **Interpretability, not causality.** Global SHAP importance describes "
    "how the trained models use features across the evaluation data. It does "
    "not prove that changing one input causes a particular lending outcome. "
    "Marital status and gender are excluded from model scoring."
)
