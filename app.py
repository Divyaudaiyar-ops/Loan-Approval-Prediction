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
    os.path.abspath(__file__)
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

ENSEMBLE_MODELS = [
    "Decision Tree",
    "Random Forest",
    "SVM",
    "XGBoost",
    "Logistic Regression",
]

render_page_hero(
    chip_label="Five-model Loan Approval Engine",
    title="Loan Approval Prediction",
    subtitle=(
        "Enter an applicant profile to get a loan approval estimate from the "
        "combined Decision Tree, Random Forest, SVM, XGBoost, and Logistic Regression models."
    ),
    mini_stats=[
        {
            "label": "Ensemble Models",
            "value": str(len(ENSEMBLE_MODELS)),
            "sub": "Equal-weight soft vote",
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
# FILE PATHS
# ============================================================

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
    load_errors = []
    model_paths = {
        "Logistic Regression": "logistic_regression.pkl",
        "Decision Tree": "decision_tree.pkl",
        "Random Forest": "random_forest.pkl",
        "SVM": "svm.pkl",
        "XGBoost": "xgboost.pkl",
        "Random Forest Tuned": "random_forest_tuned.pkl",
        "SVM Tuned": "svm_tuned.pkl",
        "XGBoost Tuned": "xgboost_tuned.pkl",
    }

    for name, filename in model_paths.items():
        path = os.path.join(MODELS_PATH, filename)
        if os.path.exists(path):
            try:
                models[name] = joblib.load(path)
            except Exception as e:
                load_errors.append(f"{name} ({filename}): {e}")

    if load_errors:
        st.warning("Some saved models could not be loaded: " + "; ".join(load_errors))

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

if not models:
    render_error_state(
        "No trained models were found in the models folder.",
        "Re-run the training pipeline to regenerate `.pkl` artifacts."
    )
    st.stop()

missing_ensemble_models = [name for name in ENSEMBLE_MODELS if name not in models]
if missing_ensemble_models:
    render_error_state(
        "The five-model prediction cannot run because these base models are unavailable: "
        + ", ".join(missing_ensemble_models),
        "Check the model load warnings and regenerate incompatible model files if needed."
    )
    st.stop()

# ============================================================
# FIVE-MODEL ENSEMBLE
# ============================================================

ensemble_name = "Five-model soft-voting ensemble"


# ============================================================
# APPLICANT FORM + LIVE SUMMARY
# ============================================================

with st.container():
    with st.form("loan_prediction_form", clear_on_submit=False):
        section_title("👤 Applicant Details")
        ac1, ac2 = st.columns(2)

        with ac1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            married = st.selectbox("Marital Status", ["Yes", "No"])

        with ac2:
            dependents = st.selectbox("Number of Dependents", [0, 1, 2, 3])
            education = st.selectbox("Education Level", ["Graduate", "Not Graduate"])

        st.markdown("")

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

        section_title("📍 Property Area")
        property_area = st.selectbox(
            "Property Location / Area Type",
            ["Semiurban", "Urban", "Rural"],
            help="Locality classification influences market risk and collateral value."
        )

        st.markdown("")

        st.divider()
        submit = st.form_submit_button("🔮 Run Loan Approval Prediction")

total_income = applicant_income + coapplicant_income
loan_to_income = loan_amount / (total_income + 1)
risk_label, risk_class, risk_note = lti_risk(loan_to_income)

st.markdown("---")

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
        with loading_spinner("Running the five-model ensemble"):
            input_data = add_features(input_data)
            model_probabilities = {
                name: float(models[name].predict_proba(input_data)[0][1])
                for name in ENSEMBLE_MODELS
            }
            probability = sum(model_probabilities.values()) / len(model_probabilities)
            prediction = int(probability >= 0.5)
    except Exception as e:
        render_error_state(
            f"Prediction failed: {str(e)}",
            "Check model compatibility with preprocessing pipeline."
        )
        st.stop()

    is_approved = prediction == 1

    agreement_df = pd.DataFrame(
        [
            {
                "Model": name,
                "Approval Probability": f"{model_probabilities[name] * 100:.2f}%",
                "Model Decision": "Approve" if model_probabilities[name] >= 0.5 else "Reject",
            }
            for name in ENSEMBLE_MODELS
        ]
    )
    section_title("🧩 Model Contributions")
    st.dataframe(agreement_df, width="stretch", hide_index=True)

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
                ("Credit History", "Good (1.0)" if credit_history == 1.0 else "Poor (0.0)"),
                ("Property Area", property_area),
                ("Loan Term (Months)", loan_term),
                ("ML Model", ensemble_name),
            ],
            columns=["Attribute", "Value"]
        )
        # Keep the mixed input values in one Arrow-compatible display type.
        snapshot = snapshot.astype({"Attribute": "string", "Value": "string"})
        st.dataframe(snapshot, width='stretch', hide_index=True, height=345)

    with full_sum2:
        st.markdown("##### Prediction & Financials")
        audit_df = pd.DataFrame(
            [
                ("Prediction Outcome", "Approved ✅" if is_approved else "Rejected ❌"),
                ("Approval Probability", f"{probability * 100:.2f}%"),
                ("Rejection Probability", f"{(1 - probability) * 100:.2f}%"),
                ("Applicant Income", f"{applicant_income:,.2f}"),
                ("Co-applicant Income", f"{coapplicant_income:,.2f}"),
                ("Total Income (calc.)", f"{total_income:,.2f}"),
                ("Loan Amount", f"{loan_amount:,.2f}"),
                ("Loan-to-Income (calc.)", f"{loan_to_income:.4f}  ({risk_label} risk)"),
                ("Predicted At", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            ],
            columns=["Attribute", "Value"]
        )
        st.dataframe(audit_df, width='stretch', hide_index=True, height=345)

    st.markdown("")

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
                ensemble_name,
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

    section_title("🎯 Prediction Result")
    result_main_col, result_gauge_col = st.columns([3, 2])

    with result_main_col:
        render_verdict_enhanced(
            approved=is_approved,
            probability=probability,
            model_name=ensemble_name,
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
    render_alert(
        "<b>Decision Support Notice</b> — This is an academic ML prototype recommendation, "
        "not a final lending decision. Always combine with manual underwriting, "
        "organisational policy checks, and regulatory compliance review.",
        kind="warning"
    )


st.markdown("---")

section_title("📋 Applicant Summary")

applicant_summary = pd.DataFrame(
    [
        ("Gender", gender),
        ("Marital Status", married),
        ("Dependents", dependents),
        ("Education", education),
        ("Self-Employed", self_employed),
        ("Credit History", "Good" if credit_history == 1.0 else "Poor"),
        ("Property Area", property_area),
    ],
    columns=["Application Profile", "Value"]
).astype("string")
st.dataframe(applicant_summary, width="stretch", hide_index=True)

st.markdown("")
section_title("🧮 Financial Calculations")

financial_columns = st.columns(2)
with financial_columns[0]:
    st.metric("Applicant Income", f"{applicant_income:,.2f}")
    st.metric("Co-applicant Income", f"{coapplicant_income:,.2f}")
    st.metric("Total Household Income", f"{total_income:,.2f}")
with financial_columns[1]:
    st.metric("Loan Amount", f"{loan_amount:,.2f}")
    st.metric("Loan Term", f"{loan_term} months ({loan_term / 12:.0f} years)")
    st.metric("Loan-to-Income Ratio", f"{loan_to_income:.4f}", delta=f"{risk_label} risk")

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

render_disclaimer_footer()
