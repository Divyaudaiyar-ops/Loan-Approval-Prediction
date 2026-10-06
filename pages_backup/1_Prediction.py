import streamlit as st
import pandas as pd
import joblib
import sqlite3
import os
import sys

from datetime import datetime


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
    render_page_hero,
    render_verdict_enhanced,
    render_probability_gauge,
    section_title,
    loading_spinner,
    render_error_state,
    risk_badge_html,
    render_alert,
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Loan Prediction · CreditFlow AI",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

inject_theme_css()
render_sidebar_brand(
    app_name="Loan Prediction",
    subtitle="ML-Driven Approval Checks"
)


# ============================================================
# FILE PATHS
# ============================================================

DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "loan_data.csv"
)

MODELS_PATH = os.path.join(
    PROJECT_ROOT,
    "models"
)

DB_PATH = os.path.join(
    PROJECT_ROOT,
    "loan_predictions.db"
)


# ============================================================
# DATABASE
# ============================================================

def create_database():

    conn = sqlite3.connect(DB_PATH)

    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            timestamp TEXT,

            model TEXT,

            prediction TEXT,

            probability REAL,

            applicant_income REAL,

            coapplicant_income REAL,

            total_income REAL,

            loan_amount REAL,

            loan_to_income REAL,

            credit_history REAL,

            education TEXT,

            property_area TEXT

        )
        """
    )

    conn.commit()
    conn.close()


create_database()


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    models = {}

    model_paths = {

        "Logistic Regression":
            "logistic_regression.pkl",

        "Decision Tree":
            "decision_tree.pkl",

        "Random Forest":
            "random_forest.pkl",

        "SVM":
            "svm.pkl",

        "XGBoost":
            "xgboost.pkl",

        "Random Forest Tuned":
            "random_forest_tuned.pkl",

        "SVM Tuned":
            "svm_tuned.pkl",

        "XGBoost Tuned":
            "xgboost_tuned.pkl"
    }

    for name, filename in model_paths.items():

        path = os.path.join(
            MODELS_PATH,
            filename
        )

        if os.path.exists(path):

            models[name] = joblib.load(path)

    return models


try:
    with loading_spinner("Warming up model library"):
        models = load_models()
except Exception as e:
    render_error_state(str(e), "Check /models folder for `.pkl` artifacts.")
    models = {}


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    section_title("📝 Loan Prediction")

    st.markdown(
        """
        Complete the applicant form, review the live summary,
        and run an ML-based approval check. All predictions
        are saved to the SQLite history log automatically.
        """
    )

    st.divider()

    st.caption("Explainable ML Based Loan Approval Prediction")
    st.caption("Academic Decision Support Prototype")


# ============================================================
# HELPER: LTI RISK LEVEL
# ============================================================

def lti_risk(lti):

    if lti < 0.02:
        return (
            "LOW",
            "risk-low",
            "Loan-to-income is comfortably below typical underwriting thresholds."
        )

    elif lti < 0.04:
        return (
            "MEDIUM",
            "risk-medium",
            "Loan-to-income is in the acceptable range but monitor closely."
        )

    else:
        return (
            "HIGH",
            "risk-high",
            "Loan-to-income is elevated and may impact approval likelihood."
        )


# ============================================================
# PAGE HERO HEADER
# ============================================================

model_descriptions = {
    "Logistic Regression": "Baseline · Interpretable coefficients · Fast",
    "Decision Tree": "Rule-based · Fully explainable splits",
    "Random Forest": "Ensemble · High accuracy · Bagging",
    "SVM": "Max-margin · Kernel · Stable",
    "XGBoost": "Gradient Boosting · SOTA tabular accuracy",
    "Random Forest Tuned": "RandomizedSearchCV tuned · Improved baseline RF",
    "SVM Tuned": "RandomizedSearchCV tuned · Improved baseline SVM",
    "XGBoost Tuned": "RandomizedSearchCV tuned · Improved baseline XGB"
}

if not models:
    render_error_state(
        "No trained models were found in the models folder.",
        "Re-run the training pipeline to regenerate `.pkl` artifacts."
    )
    st.stop()

render_page_hero(
    chip_label="Single-Run Prediction Engine",
    title="Loan Approval Prediction",
    subtitle=(
        "Complete the application form to obtain a machine-learning driven loan approval "
        "decision. Inputs are validated in real time, a live Applicant Summary is produced, "
        "and the outcome is persisted to SQLite for auditing."
    ),
    mini_stats=[
        {
            "label": "Available Models",
            "value": str(len(models)),
            "sub": "5 Base · 3 Tuned",
            "variant": "primary"
        },
        {
            "label": "Decision Threshold",
            "value": "50.0%",
            "sub": "Probability ≥ 50% → Approved",
            "variant": "info"
        },
        {
            "label": "Auto-Save",
            "value": "On",
            "sub": "SQLite history log",
            "variant": "success"
        },
        {
            "label": "Explainability",
            "value": "SHAP",
            "sub": "Global + Local (XAI page)",
            "variant": "default"
        },
    ]
)


# ============================================================
# APPLICANT FORM + LIVE SUMMARY
# ============================================================

form_col, summary_col = st.columns([3, 2])

# ---------- FORM ----------

with form_col:

    with st.form(
        "loan_prediction_form",
        clear_on_submit=False
    ):

        # 1. Applicant Details
        section_title("👤 Applicant Details")

        ac1, ac2 = st.columns(2)

        with ac1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            married = st.selectbox("Marital Status", ["Yes", "No"])

        with ac2:
            dependents = st.selectbox("Number of Dependents", [0, 1, 2, 3])
            education = st.selectbox("Education Level", ["Graduate", "Not Graduate"])

        st.markdown("")

        # 2. Income and Financial Details
        section_title("💰 Income and Financial Details")

        fi1, fi2 = st.columns(2)

        with fi1:
            self_employed = st.selectbox("Self Employed", ["No", "Yes"])

        st.markdown("")

        inc1, inc2 = st.columns(2)

        with inc1:
            applicant_income = st.number_input(
                "Applicant Income",
                min_value=0.0,
                value=5000.0,
                step=100.0,
                format="%.2f",
                help="Primary applicant's monthly/annual gross income."
            )

        with inc2:
            coapplicant_income = st.number_input(
                "Co-applicant Income",
                min_value=0.0,
                value=0.0,
                step=100.0,
                format="%.2f",
                help="Secondary income (spouse, guarantor, etc.). 0 if none."
            )

        st.markdown("")

        # 3. Credit History
        section_title("💳 Credit History")

        credit_history = st.selectbox(
            "Credit History (Repaid previous debts / meets credit guidelines)",
            [1.0, 0.0],
            format_func=lambda v: (
                "✅ Good – Meets credit underwriting criteria (1.0)"
                if v == 1.0
                else "❌ Poor / Absent – Does not meet criteria (0.0)"
            ),
            help="The single strongest predictor of loan approval in most credit scoring models."
        )

        st.markdown("")

        # 4. Loan Amount and Term
        section_title("🏦 Loan Amount and Term")

        ln1, ln2 = st.columns(2)

        with ln1:
            loan_amount = st.number_input(
                "Loan Amount (in thousands)",
                min_value=0.0,
                value=150.0,
                step=10.0,
                format="%.2f",
                help="Requested principal loan amount."
            )

        with ln2:
            loan_term = st.selectbox(
                "Loan Amount Term (months)",
                [360, 300, 240, 180, 120, 84, 60, 36],
                format_func=lambda v: (
                    f"{v} months ({v/12:.0f} years)"
                    if v >= 12 else f"{v} months"
                ),
                help="Repayment period in months. Standard = 360 (30yr)."
            )

        st.markdown("")

        # 5. Property Area
        section_title("📍 Property Area")

        property_area = st.selectbox(
            "Property Location / Area Type",
            ["Semiurban", "Urban", "Rural"],
            help="Locality classification influences market risk and collateral value."
        )

        st.markdown("")

        # 6. Model Selection
        section_title("🤖 Model Selection")

        model_name = st.selectbox(
            "Machine Learning Model",
            list(models.keys()),
            format_func=lambda n: f"{n}  —  {model_descriptions.get(n, '')}"
        )

        st.divider()

        submit = st.form_submit_button(
            "🔮 Run Loan Approval Prediction"
        )


# ---------- LIVE APPLICANT SUMMARY ----------

with summary_col:

    section_title("📋 Applicant Summary")

    total_income = applicant_income + coapplicant_income
    loan_to_income = loan_amount / (total_income + 1)
    risk_label, risk_class, risk_note = lti_risk(loan_to_income)

    st.markdown(
        f"""
        <div class="summary-card">
            <div style="margin-bottom:14px;">
                <strong style="font-size:15px;color:#0f172a;">
                    📌 Application Profile
                </strong>
            </div>

            <div class="summary-row">
                <span class="summary-label">Gender</span>
                <span class="summary-value">{gender}</span>
            </div>
            <div class="summary-row">
                <span class="summary-label">Marital Status</span>
                <span class="summary-value">{married}</span>
            </div>
            <div class="summary-row">
                <span class="summary-label">Dependents</span>
                <span class="summary-value">{dependents}</span>
            </div>
            <div class="summary-row">
                <span class="summary-label">Education</span>
                <span class="summary-value">{education}</span>
            </div>
            <div class="summary-row">
                <span class="summary-label">Self-Employed</span>
                <span class="summary-value">{self_employed}</span>
            </div>
            <div class="summary-row">
                <span class="summary-label">Credit History</span>
                <span class="summary-value">
                    {'✅ Good' if credit_history == 1.0 else '❌ Poor'}
                </span>
            </div>
            <div class="summary-row">
                <span class="summary-label">Property Area</span>
                <span class="summary-value">{property_area}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("")

    section_title("🧮 Financial Calculations")

    st.markdown(
        f"""
        <div class="summary-card">
            <div style="margin-bottom:14px;">
                <strong style="font-size:15px;color:#0f172a;">
                    💰 Derived Metrics (Live)
                </strong>
            </div>

            <div class="summary-row">
                <span class="summary-label">Applicant Income</span>
                <span class="summary-value">{applicant_income:,.2f}</span>
            </div>
            <div class="summary-row">
                <span class="summary-label">Co-applicant Income</span>
                <span class="summary-value">{coapplicant_income:,.2f}</span>
            </div>
            <div class="summary-row highlight">
                <span class="summary-label" style="font-weight:700;">📊 Total Income</span>
                <span class="summary-value" style="font-size:15px;color:#1e40af;">
                    {total_income:,.2f}
                </span>
            </div>
            <div class="summary-row">
                <span class="summary-label">Loan Amount</span>
                <span class="summary-value">{loan_amount:,.2f}</span>
            </div>
            <div class="summary-row">
                <span class="summary-label">Loan Term</span>
                <span class="summary-value">{loan_term} months ({loan_term/12:.0f} yr)</span>
            </div>
            <div class="summary-row highlight" style="background:#fffbeb;">
                <span class="summary-label" style="font-weight:700;">
                    📐 Loan-to-Income (LTI)
                </span>
                <span>
                    <span class="summary-value" style="font-size:15px;">
                        {loan_to_income:.4f}
                    </span>
                    &nbsp;
                    {risk_badge_html(risk_label)}
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    with st.expander("ℹ️  Understanding Loan-to-Income Ratio", expanded=False):
        st.markdown(
            f"""
            **LTI = Loan Amount / Total Income = {loan_to_income:.4f}**

            - **Below 0.02 (LOW):** Comfortable affordability.
            - **0.02 – 0.04 (MEDIUM):** Standard acceptable range.
            - **Above 0.04 (HIGH):** Red-flag territory; repayment burden may be excessive.

            {risk_note}
            """
        )

    st.markdown("")

    section_title("🤖 Selected Model")

    render_alert(
        f"<b>{model_name}</b><br><span style='font-size:13px;'>{model_descriptions.get(model_name, '')}</span>",
        kind="info"
    )


# ============================================================
# PREDICTION RESULT
# ============================================================

if submit:

    input_data = pd.DataFrame(
        {
            "Gender": [gender],
            "Married": [married],
            "Dependents": [dependents],
            "Education": [education],
            "Self_Employed": [self_employed],
            "ApplicantIncome": [applicant_income],
            "CoapplicantIncome": [coapplicant_income],
            "LoanAmount": [loan_amount],
            "Loan_Amount_Term": [loan_term],
            "Credit_History": [credit_history],
            "Property_Area": [property_area]
        }
    )

    try:
        with loading_spinner(f"Running inference with {model_name}"):
            input_data = add_features(input_data)
            selected_model = models[model_name]
            prediction = selected_model.predict(input_data)[0]
            probability = selected_model.predict_proba(input_data)[0][1]
    except Exception as e:
        render_error_state(
            f"Prediction failed: {str(e)}",
            "Check model compatibility with preprocessing pipeline."
        )
        st.stop()

    is_approved = prediction == 1

    section_title("🎯 Prediction Result")

    # ---- Enhanced verdict banner + prob gauge ----
    result_main_col, result_gauge_col = st.columns([3, 2])

    with result_main_col:
        render_verdict_enhanced(
            approved=is_approved,
            probability=probability,
            model_name=model_name,
            total_income=total_income,
            loan_amount=loan_amount,
            lti=loan_to_income,
            credit_good=(credit_history == 1.0),
            subtitle=(
                "Model recommendation based on the submitted applicant profile and "
                "learned patterns from historical loan applications."
            )
        )

    with result_gauge_col:
        render_probability_gauge(
            probability=probability,
            threshold=0.5,
            title="Approval Probability Spectrum",
            show_threshold=True
        )

    st.markdown("")

    # ---- Enhanced result summary KPIs ----
    section_title("🧾 Key Financial & Decision KPIs")

    det1, det2, det3, det4 = st.columns(4)

    with det1:
        st.metric(
            "Total Household Income",
            f"{total_income:,.2f}",
            help="Applicant + Co-applicant income"
        )

    with det2:
        st.metric(
            "Requested Loan Amount",
            f"{loan_amount:,.2f}",
            f"{loan_term/12:.0f}-year term"
        )

    with det3:
        st.metric(
            "Loan-to-Income Ratio",
            f"{loan_to_income:.4f}",
            f"RISK: {risk_label}"
        )

    with det4:
        threshold_delta = (probability - 0.5) * 100
        st.metric(
            "Probability vs Threshold",
            f"{threshold_delta:+.1f} pp",
            "vs 50% decision line"
        )

    st.markdown("")

    # ---- Full applicant summary on result ----
    section_title("📋 Complete Audit Snapshot")

    full_sum1, full_sum2 = st.columns(2)

    with full_sum1:

        st.markdown("##### Applicant Profile")

        snapshot = pd.DataFrame(
            [
                ("Gender", gender),
                ("Marital Status", married),
                ("Dependents", dependents),
                ("Education", education),
                ("Self Employed", self_employed),
                (
                    "Credit History",
                    "Good (1.0)" if credit_history == 1.0 else "Poor (0.0)"
                ),
                ("Property Area", property_area),
                ("Loan Term (Months)", loan_term),
                ("ML Model", model_name),
            ],
            columns=["Attribute", "Value"]
        )

        st.dataframe(
            snapshot,
            width='stretch',
            hide_index=True,
            height=345
        )

    with full_sum2:

        st.markdown("##### Prediction & Financials")

        audit_df = pd.DataFrame(
            [
                (
                    "Prediction Outcome",
                    "Approved ✅" if is_approved else "Rejected ❌"
                ),
                ("Approval Probability", f"{probability * 100:.2f}%"),
                (
                    "Rejection Probability",
                    f"{(1 - probability) * 100:.2f}%"
                ),
                ("Applicant Income", f"{applicant_income:,.2f}"),
                ("Co-applicant Income", f"{coapplicant_income:,.2f}"),
                ("Total Income (calc.)", f"{total_income:,.2f}"),
                ("Loan Amount", f"{loan_amount:,.2f}"),
                (
                    "Loan-to-Income (calc.)",
                    f"{loan_to_income:.4f}  ({risk_label} risk)"
                ),
                (
                    "Predicted At",
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                ),
            ],
            columns=["Attribute", "Value"]
        )

        st.dataframe(
            audit_df,
            width='stretch',
            hide_index=True,
            height=345
        )

    st.markdown("")

    # ---- Save prediction ----
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO predictions
            (
                timestamp,
                model,
                prediction,
                probability,
                applicant_income,
                coapplicant_income,
                total_income,
                loan_amount,
                loan_to_income,
                credit_history,
                education,
                property_area
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                model_name,
                "Approved" if is_approved else "Rejected",
                float(probability),
                applicant_income,
                coapplicant_income,
                total_income,
                loan_amount,
                loan_to_income,
                credit_history,
                education,
                property_area
            )
        )

        conn.commit()
        conn.close()

        render_alert(
            "✅ Prediction saved to history. Navigate to the **Prediction History** page in the "
            "sidebar to review past runs, filter records, and export as CSV.",
            kind="success"
        )
    except Exception as e:
        render_error_state(
            f"Could not save prediction to history: {str(e)}",
            "Check SQLite file permissions at `loan_predictions.db`."
        )

    st.markdown("")

    render_alert(
        "<b>Decision Support Notice</b> — This is an academic ML prototype recommendation, "
        "not a final lending decision. Always combine with manual underwriting, "
        "organisational policy checks, and regulatory compliance review.",
        kind="warning"
    )


# ============================================================
# ACADEMIC DISCLAIMER FOOTER
# ============================================================

render_disclaimer_footer()
