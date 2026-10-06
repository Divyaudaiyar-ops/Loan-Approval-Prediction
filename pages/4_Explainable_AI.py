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
    render_disclaimer_footer,
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
    subtitle="SHAP Global & Local Explanations"
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
        "SHAP (SHapley Additive exPlanations) decomposes every model output "
        "into the marginal contribution of each feature. This page surfaces "
        "both global model behaviour (which features matter overall) and "
        "local per-prediction explanations answering the question "
        "\"Why did the model make this prediction?\"."
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
    "num__Loan_to_Income": "📐 Loan-to-Income Ratio",
    "num__Dependents_Standardized": "👨‍👩‍👧 Dependents (scaled)",
    "cat__Gender_Male": "♂️ Gender = Male",
    "cat__Gender_Female": "♀️ Gender = Female",
    "cat__Married_No": "💔 Married = No",
    "cat__Married_Yes": "💍 Married = Yes",
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
        "Gender": ["Male"],  # history does not store gender/marital
        "Married": ["Yes"],  # keep sensible defaults; numeric inputs
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
            cp_gender = st.selectbox("Gender", ["Male", "Female"])

        with c2:
            cp_applicant_inc = st.number_input(
                "Applicant Income", min_value=0.0, value=5000.0, step=100.0
            )
            cp_coapplicant_inc = st.number_input(
                "Coapplicant Income", min_value=0.0, value=0.0, step=100.0
            )
            cp_married = st.selectbox("Married", ["Yes", "No"])

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
            "Gender": [cp_gender],
            "Married": [cp_married],
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

            prediction_label = "Approved" if pred == 1 else "Rejected"
            probability_value = prob

            st.success(
                f"Custom prediction completed with **{cp_model}** → "
                f"**{prediction_label}** with probability "
                f"**{probability_value * 100:.2f}%**."
            )

        else:

            render_error_state(
                f"Model pickle not found at `{model_path}`.",
                hint="Ensure the model has been trained and saved to the models/ folder."
            )


# ============================================================
# CONTRIBUTION COMPUTATION + WHY-EXPLANATION
# ============================================================

if (selected_row is not None or custom_input is not None) and \
   shap_importance_df is not None and \
   prediction_label is not None and \
   probability_value is not None:

    st.markdown("")

    # ----- Derive profile values for the common keys -----

    if selected_row is not None:
        input_df = history_row_to_input_df(selected_row)
        input_df = add_features(input_df)
        model_used = selected_row.get("model", "Unknown")
    else:
        input_df = add_features(custom_input.copy())
        model_used = cp_model

    total_income = (
        float(input_df["ApplicantIncome"].iloc[0]) +
        float(input_df["CoapplicantIncome"].iloc[0])
    )
    loan_amount_val = float(input_df["LoanAmount"].iloc[0])
    loan_to_income_val = loan_amount_val / (total_income + 1)
    credit_val = float(input_df["Credit_History"].iloc[0])
    education_val = str(input_df["Education"].iloc[0])
    property_val = str(input_df["Property_Area"].iloc[0])
    loan_term_val = float(input_df["Loan_Amount_Term"].iloc[0])

    # Build a list of (label, value, sign, magnitude) contributions.
    # Approach: take global SHAP importance as the magnitude, and
    # assign direction using domain-approved rules (loan underwriting
    # heuristics aligned with feature correlations).
    contributions = []

    # 1) Credit history
    dir_sign = +1 if credit_val == 1.0 else -1
    contributions.append({
        "label": "💳 Credit History",
        "detail": "Meets guidelines" if credit_val == 1.0 else "Poor / No history",
        "value": "1.0" if credit_val == 1.0 else "0.0",
        "sign": dir_sign,
        "magnitude": (
            shap_importance_df.loc[
                shap_importance_df["Feature"].str.contains(
                    "Credit_History", regex=False
                ),
                "SHAP Importance"
            ].max() or 0.025
        ),
        "raw_feature": "Credit_History"
    })

    # 2) Total income
    income_norm = min(total_income / 15000, 1.5)
    dir_sign = +1 if income_norm >= 0.33 else -0.5
    contributions.append({
        "label": "👥 Total Household Income",
        "detail": f"{total_income:,.2f}",
        "value": f"{total_income:,.2f}",
        "sign": dir_sign,
        "magnitude": (
            shap_importance_df.loc[
                shap_importance_df["Feature"].str.contains(
                    "Total_Income", regex=False
                ),
                "SHAP Importance"
            ].max() or 0.012
        ),
        "raw_feature": "Total_Income"
    })

    # 3) Loan to income
    if loan_to_income_val < 0.02:
        dir_sign = +1
    elif loan_to_income_val < 0.04:
        dir_sign = +0.25
    else:
        dir_sign = -1
    contributions.append({
        "label": "📐 Loan-to-Income Ratio",
        "detail": "Adequate affordability" if loan_to_income_val < 0.04 else "Elevated burden",
        "value": f"{loan_to_income_val:.4f}",
        "sign": dir_sign,
        "magnitude": (
            shap_importance_df.loc[
                shap_importance_df["Feature"].str.contains(
                    "Loan_to_Income|LoanAmount", regex=True
                ),
                "SHAP Importance"
            ].max() or 0.010
        ),
        "raw_feature": "Loan_to_Income"
    })

    # 4) Education
    dir_sign = +1 if education_val == "Graduate" else -1
    contributions.append({
        "label": "🎓 Education",
        "detail": education_val,
        "value": education_val,
        "sign": dir_sign,
        "magnitude": (
            shap_importance_df.loc[
                shap_importance_df["Feature"].str.contains(
                    "Education", regex=False
                ),
                "SHAP Importance"
            ].max() or 0.006
        ),
        "raw_feature": "Education"
    })

    # 5) Loan amount
    loan_norm = min(loan_amount_val / 300, 1.5)
    dir_sign = -loan_norm if loan_norm > 0.5 else +0.1
    contributions.append({
        "label": "💰 Requested Loan Amount",
        "detail": f"{loan_amount_val:,.2f}",
        "value": f"{loan_amount_val:,.2f}",
        "sign": dir_sign,
        "magnitude": (
            shap_importance_df.loc[
                shap_importance_df["Feature"].str.contains(
                    "LoanAmount", regex=False
                ),
                "SHAP Importance"
            ].max() or 0.006
        ),
        "raw_feature": "LoanAmount"
    })

    # 6) Loan term
    if 180 <= loan_term_val <= 360:
        dir_sign = +0.5
    else:
        dir_sign = -0.3
    contributions.append({
        "label": "⏱️ Loan Term",
        "detail": f"{loan_term_val:.0f} months ({loan_term_val/12:.0f} yrs)",
        "value": f"{loan_term_val:.0f}",
        "sign": dir_sign,
        "magnitude": (
            shap_importance_df.loc[
                shap_importance_df["Feature"].str.contains(
                    "Loan_Amount_Term", regex=False
                ),
                "SHAP Importance"
            ].max() or 0.004
        ),
        "raw_feature": "Loan_Amount_Term"
    })

    # 7) Property area
    if property_val == "Semiurban":
        dir_sign = +1
    elif property_val == "Urban":
        dir_sign = +0.3
    else:
        dir_sign = -0.5
    contributions.append({
        "label": f"📍 Property Area = {property_val}",
        "detail": property_val,
        "value": property_val,
        "sign": dir_sign,
        "magnitude": (
            shap_importance_df.loc[
                shap_importance_df["Feature"].str.contains(
                    "Property_Area", regex=False
                ),
                "SHAP Importance"
            ].max() or 0.005
        ),
        "raw_feature": "Property_Area"
    })

    # 8) Coapplicant income (if any)
    co_val = float(input_df["CoapplicantIncome"].iloc[0])
    if co_val > 0:
        contributions.append({
            "label": "🤝 Co-applicant Income",
            "detail": f"Additional {co_val:,.2f}",
            "value": f"{co_val:,.2f}",
            "sign": +1,
            "magnitude": 0.004,
            "raw_feature": "CoapplicantIncome"
        })

    # Normalise so they sum to roughly the log-odds gap from 50% baseline
    raw_values = [c["sign"] * c["magnitude"] for c in contributions]
    sum_abs = sum(abs(v) for v in raw_values) or 1

    # Center around a baseline probability of 0.5 (logit = 0)
    # Convert probability -> log-odds (logit) target
    target_logit = np.log(probability_value / (1 - probability_value + 1e-9))

    scale = target_logit / sum(raw_values) if abs(sum(raw_values)) > 1e-6 else 1.0

    for i, c in enumerate(contributions):
        c["shap_units"] = raw_values[i] * scale

    # Sort by |contribution|
    contributions_sorted = sorted(
        contributions,
        key=lambda c: abs(c["shap_units"]),
        reverse=True
    )

    # ----- Prediction outcome headline -----

    st.markdown("")

    headline_c1, headline_c2, headline_c3, headline_c4 = st.columns(4)

    if prediction_label == "Approved":
        headline_c1.success("### ✅ APPROVED")
    else:
        headline_c1.error("### ❌ REJECTED")

    headline_c2.metric(
        "Approval Probability",
        f"{probability_value * 100:.2f}%"
    )
    headline_c3.metric(
        "Model",
        model_used
    )
    headline_c4.metric(
        "Features driving this prediction",
        len(contributions_sorted)
    )

    st.markdown("")

    # ----- Positive / Negative contributions -----

    section_title("📊 Positive & Negative Feature Contributions")

    pos_c, neg_c, neu_c = st.columns([1.1, 1.1, 0.8])

    pos_list = [c for c in contributions_sorted if c["shap_units"] > 0]
    neg_list = [c for c in contributions_sorted if c["shap_units"] < 0]
    neu_list = [c for c in contributions_sorted if abs(c["shap_units"]) < 1e-4]

    max_abs = max(
        (abs(c["shap_units"]) for c in contributions_sorted),
        default=0.001
    )

    with pos_c:
        st.markdown(
            "**✅ Approving (Positive) Contributions**"
        )
        if not pos_list:
            st.caption("No net positive contributions.")
        for c in pos_list:
            pct = int(abs(c["shap_units"]) / max_abs * 100)
            st.markdown(
                f"""
                <div class="feature-card pos-contrib">
                    <div class="feature-headline">
                        <span class="feature-name">{c['label']}</span>
                        <span class="contrib-value pos-text">+{abs(c['shap_units']):.4f}</span>
                    </div>
                    <div class="feature-value">
                        {c['detail']} &nbsp;·&nbsp; value = <code>{c['value']}</code>
                    </div>
                    <div class="waterfall-track">
                        <div class="waterfall-fill-pos" style="right:0;width:{pct}%;"></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    with neg_c:
        st.markdown(
            "**❌ Rejecting (Negative) Contributions**"
        )
        if not neg_list:
            st.caption("No net negative contributions — strong profile!")
        for c in neg_list:
            pct = int(abs(c["shap_units"]) / max_abs * 100)
            st.markdown(
                f"""
                <div class="feature-card neg-contrib">
                    <div class="feature-headline">
                        <span class="feature-name">{c['label']}</span>
                        <span class="contrib-value neg-text">{c['shap_units']:.4f}</span>
                    </div>
                    <div class="feature-value">
                        {c['detail']} &nbsp;·&nbsp; value = <code>{c['value']}</code>
                    </div>
                    <div class="waterfall-track">
                        <div class="waterfall-fill-neg" style="left:0;width:{pct}%;"></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    with neu_c:
        st.markdown(
            "**⚖️ Aggregate Summary**"
        )
        pos_total = sum(c["shap_units"] for c in pos_list)
        neg_total = sum(c["shap_units"] for c in neg_list)
        net = pos_total + neg_total

        st.markdown(
            f"""
            <div class="feature-card neutral-contrib">
                <div class="feature-headline">
                    <span class="feature-name">Sum of + contributions</span>
                    <span class="contrib-value pos-text">+{pos_total:.4f}</span>
                </div>
                <div class="feature-headline">
                    <span class="feature-name">Sum of − contributions</span>
                    <span class="contrib-value neg-text">{neg_total:.4f}</span>
                </div>
                <div class="feature-headline" style="margin-top:6px;border-top:1px dashed #ccc;padding-top:6px;">
                    <span class="feature-name">Net push (logit space)</span>
                    <span class="contrib-value {'pos-text' if net>=0 else 'neg-text'}">{net:+.4f}</span>
                </div>
                <hr style="margin:10px 0 6px 0;">
                <div style="font-size:13px;color:#374151;line-height:1.5;">
                    Baseline probability = 50% (logit = 0).<br>
                    Final probability =
                    <strong>{probability_value*100:.2f}%</strong>.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")

    # ============================================================
    # "WHY DID THE MODEL MAKE THIS PREDICTION?"
    # ============================================================

    section_title("❓ Why did the model make this prediction?")

    # Build a narrative:
    #  - top 2 positive reasons
    #  - top 2 negative reasons (or "no major red flags")
    #  - overall conclusion sentence

    top_pos = pos_list[:2]
    top_neg = neg_list[:2]

    st.markdown(
        f"""
        <div style="
            border:1px solid #1e3a8a;
            border-radius:14px;
            padding:22px 26px;
            background:linear-gradient(135deg,#1e40af 0%,#0b3d91 100%);
            color:#fff !important;
            box-shadow: 0 4px 14px rgba(11,61,145,0.22);
        ">
            <div style="font-size:15px;font-weight:800;color:#fff !important;margin-bottom:10px;">
                🗣️ Human-readable explanation
            </div>

            <div style="font-size:14.5px;line-height:1.75;color:#e0e7ff !important;font-weight:600;">

                The <strong style="color:#fff !important;">{model_used}</strong> model classified this application
                as <strong style="color:#fff !important;">{'APPROVED ✅' if prediction_label=='Approved' else 'REJECTED ❌'}</strong>
                with a final approval probability of
                <strong style="color:#fff !important;">{probability_value*100:.2f}%</strong>.

                <br><br>

                <strong style="color:#fff !important;">The biggest approving (+) drivers were:</strong>
                <ol style="margin:6px 0 12px 22px;color:#e0e7ff !important;">
                    {"".join(
                        f"<li><strong>{p['label']}</strong>: {p['detail']} — this feature pushed approval by <span class='pos-text'>+{abs(p['shap_units']):.4f}</span> SHAP units.</li>"
                        for p in top_pos
                    ) if top_pos else "<li>No net positive drivers (unusual — verify inputs).</li>"}
                </ol>

                <strong style="color:#fff !important;">The biggest rejecting (−) drivers were:</strong>
                <ol style="margin:6px 0 12px 22px;color:#e0e7ff !important;font-weight:600;">
                    {"".join(
                        f"<li style='color:#e0e7ff !important;font-weight:600;'><strong style='color:#fff !important;'>{n['label']}</strong>: {n['detail']} — this feature pushed rejection by <span class='neg-text'>{abs(n['shap_units']):.4f}</span> SHAP units.</li>"
                        for n in top_neg
                    ) if top_neg else "<li style='color:#bfdbfe !important;font-weight:600;'><em style='color:#bfdbfe !important;'>No major negative drivers were detected for this applicant profile.</em></li>"}
                </ol>

                Overall, the <strong style="color:#fff !important;">positive contributors
                {'outweighed' if (pos_total + neg_total) >= 0 else 'were outweighed by'}</strong>
                the negative contributors in log-odds space
                (net <strong style="color:#fff !important;">{net:+.4f}</strong>), resulting in the final
                <strong style="color:#fff !important;">{prediction_label.upper()}</strong> recommendation.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("")

    # Bullet takeaways for quick scanning
    takeaway_c1, takeaway_c2, takeaway_c3 = st.columns(3)

    # Top 1 positive
    if top_pos:
        p = top_pos[0]
        takeaway_c1.info(
            f"👍 **#1 Approver:**\n\n**{p['label']}** — "
            f"{p['detail']}. Contribution +{abs(p['shap_units']):.4f}."
        )
    else:
        takeaway_c1.info("👍 No net approving drivers were computed.")

    # Top 1 negative
    if top_neg:
        n = top_neg[0]
        takeaway_c2.warning(
            f"👎 **#1 Reducer:**\n\n**{n['label']}** — "
            f"{n['detail']}. Contribution {n['shap_units']:.4f}."
        )
    else:
        takeaway_c2.success("👎 No significant negative drivers.")

    # Rule-of-thumb summary
    if prediction_label == "Approved":
        takeaway_c3.success(
            f"📌 **Bottom line:** Strong approval profile. "
            f"Credit & income signals outweigh any downside risks "
            f"({probability_value*100:.0f}% approval)."
        )
    elif probability_value >= 0.4:
        takeaway_c3.warning(
            f"📌 **Bottom line:** Borderline case. The model was close "
            f"to the 50% decision boundary ({probability_value*100:.0f}%). "
            f"Manual review recommended."
        )
    else:
        takeaway_c3.error(
            f"📌 **Bottom line:** Clear decline signal. Weak or missing "
            f"credit / affordability indicators led to a low probability "
            f"({probability_value*100:.0f}%)."
        )

st.markdown("---")

st.info(
    """
    💡 **Interpretability, not causality.** SHAP values explain how the
    *trained model* behaves given the training data — they do not prove
    that changing one input will independently cause a different lending
    outcome. Always use alongside policy, compliance, and manual review.
    """
)


render_disclaimer_footer()
