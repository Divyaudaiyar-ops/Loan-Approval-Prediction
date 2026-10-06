import streamlit as st
import pandas as pd
import joblib
import sqlite3
import os
import sys
import math

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
    render_page_hero,
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
            "value": "3 outcomes",
            "sub": "Approve · Review · Not recommended",
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
            property_area TEXT,
            marital_status TEXT,
            dependents INTEGER,
            self_employed TEXT,
            loan_term_months INTEGER,
            annual_interest_rate REAL,
            existing_monthly_debt REAL,
            estimated_monthly_payment REAL,
            debt_to_income REAL,
            decision_explanation TEXT,
            decision_basis TEXT
        )
        """
    )
    extra_columns = {
        "marital_status": "TEXT",
        "dependents": "INTEGER",
        "self_employed": "TEXT",
        "loan_term_months": "INTEGER",
        "annual_interest_rate": "REAL",
        "existing_monthly_debt": "REAL",
        "estimated_monthly_payment": "REAL",
        "debt_to_income": "REAL",
        "decision_explanation": "TEXT",
        "decision_basis": "TEXT",
    }
    existing_columns = {
        row[1] for row in cursor.execute("PRAGMA table_info(predictions)")
    }
    for column, column_type in extra_columns.items():
        if column not in existing_columns:
            cursor.execute(
                f"ALTER TABLE predictions ADD COLUMN {column} {column_type}"
            )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            prediction_id INTEGER NOT NULL UNIQUE,
            created_at TEXT NOT NULL,
            gender TEXT,
            marital_status TEXT,
            dependents INTEGER,
            education TEXT,
            self_employed TEXT,
            applicant_income REAL,
            coapplicant_income REAL,
            total_income REAL,
            credit_history REAL,
            property_area TEXT,
            loan_amount REAL,
            loan_term_months INTEGER,
            estimated_monthly_payment REAL,
            existing_monthly_debt REAL,
            debt_to_income REAL,
            approval_status TEXT NOT NULL,
            approval_probability REAL,
            decision_basis TEXT,
            FOREIGN KEY (prediction_id) REFERENCES predictions(id)
        )
        """
    )
    cursor.execute(
        """
        INSERT OR IGNORE INTO users
        (
            prediction_id, created_at, marital_status, dependents, education,
            self_employed, applicant_income, coapplicant_income, total_income,
            credit_history, property_area, loan_amount, loan_term_months,
            estimated_monthly_payment, existing_monthly_debt, debt_to_income,
            approval_status, approval_probability, decision_basis
        )
        SELECT
            id, timestamp, marital_status, dependents, education, self_employed,
            applicant_income, coapplicant_income, total_income, credit_history,
            property_area, loan_amount, loan_term_months,
            estimated_monthly_payment, existing_monthly_debt, debt_to_income,
            prediction, probability,
            COALESCE(decision_basis, decision_explanation,
                     'Decision basis was not captured for this older prediction.')
        FROM predictions
        """
    )
    old_review_rows = cursor.execute(
        """
        SELECT id, probability, debt_to_income, credit_history, property_area
        FROM predictions
        WHERE prediction = 'Manual Review'
        """
    ).fetchall()
    for prediction_id, score, dti, credit, area in old_review_rows:
        score_pct = f"{float(score) * 100:.2f}%" if score is not None else "unavailable"
        dti_pct = f"{float(dti) * 100:.2f}%" if dti is not None else "unavailable"
        credit_text = "Good" if credit == 1.0 else "Poor / not established"
        basis = (
            f"Status: Rejected under the binary policy. Stored five-model average: {score_pct}; "
            f"approval requires at least 60% and DTI at or below 36%. Stored DTI: {dti_pct}. "
            f"Credit history: {credit_text}. Property area: {area or 'unavailable'}."
        )
        explanation = (
            "Rejected under the binary policy because the application did not satisfy "
            "both the model-score and affordability requirements."
        )
        cursor.execute(
            "UPDATE predictions SET prediction = 'Rejected', decision_explanation = ?, decision_basis = ? WHERE id = ?",
            (explanation, basis, prediction_id)
        )
        cursor.execute(
            "UPDATE users SET approval_status = 'Rejected', decision_basis = ? WHERE prediction_id = ?",
            (basis, prediction_id)
        )
    conn.commit()
    conn.close()


create_database()


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models(model_signature):
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
        model_filenames = [
            "logistic_regression.pkl",
            "decision_tree.pkl",
            "random_forest.pkl",
            "svm.pkl",
            "xgboost.pkl",
            "random_forest_tuned.pkl",
            "svm_tuned.pkl",
            "xgboost_tuned.pkl",
        ]
        model_signature = tuple(
            (filename, os.path.getmtime(os.path.join(MODELS_PATH, filename)),
             os.path.getsize(os.path.join(MODELS_PATH, filename)))
            for filename in model_filenames
            if os.path.exists(os.path.join(MODELS_PATH, filename))
        )
        models = load_models(model_signature)
except Exception as e:
    render_error_state(str(e), "Check /models folder for `.pkl` artifacts.")
    models = {}


# ============================================================
# SIDEBAR
# ============================================================

# ============================================================
# HELPER: LTI RISK LEVEL
# ============================================================

def calculate_monthly_payment(principal, annual_rate_percent, term_months):
    """Calculate a standard fixed-rate monthly principal-and-interest payment."""
    monthly_rate = annual_rate_percent / 1200.0
    if monthly_rate == 0:
        return principal / term_months
    growth = (1 + monthly_rate) ** term_months
    return principal * monthly_rate * growth / (growth - 1)


def affordability_band(debt_to_income):
    """Illustrative prototype bands; lenders set product-specific limits."""
    if debt_to_income <= 0.36:
        return "Within demo limit", "Monthly obligations are within the prototype's 36% approval limit."
    if debt_to_income <= 0.43:
        return "Above demo limit", "Monthly obligations exceed the prototype's 36% approval limit."
    return "High", "Monthly obligations exceed the prototype's approval limit by a wide margin."


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
            married = st.selectbox(
                "Marital Status",
                ["Yes", "No"],
                help="Recorded for applicant context; marital status alone does not change the recommendation."
            )

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
                "Applicant Gross Monthly Income",
                min_value=0.0,
                value=5000.0,
                step=100.0,
                format="%.2f",
                help="Use gross monthly income in the same units as the training data."
            )

        with inc2:
            coapplicant_income = st.number_input(
                "Co-applicant Gross Monthly Income",
                min_value=0.0,
                value=0.0,
                step=100.0,
                format="%.2f",
                help="Verified co-applicant gross monthly income in the same units. Enter 0 if none."
            )

        existing_monthly_debt = st.number_input(
            "Existing Monthly Debt Payments",
            min_value=0.0,
            value=0.0,
            step=50.0,
            format="%.2f",
            help="Include recurring loan, credit-card, and other debt payments."
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
        ln1, ln2, ln3 = st.columns(3)

        with ln1:
            loan_amount = st.number_input(
                "Loan Amount (in thousands)",
                min_value=0.0,
                value=150.0,
                step=10.0,
                format="%.2f",
                help="Enter the principal in thousands; for example, 150 means 150,000 currency units."
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

        with ln3:
            annual_interest_rate = st.number_input(
                "Annual Interest Rate (%)",
                min_value=0.0,
                max_value=50.0,
                value=8.0,
                step=0.25,
                format="%.2f",
                help="Use the rate quoted for this loan. The 8% starting value is only a demo assumption."
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
loan_principal = loan_amount * 1000
estimated_monthly_payment = calculate_monthly_payment(
    loan_principal, annual_interest_rate, loan_term
)
monthly_obligations = estimated_monthly_payment + existing_monthly_debt
debt_to_income = (
    monthly_obligations / total_income
    if total_income > 0 else math.inf
)
affordability_status, affordability_note = affordability_band(debt_to_income)

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
            model_input = input_data.drop(columns=["Gender", "Married"])
            model_input = add_features(model_input)
            model_probabilities = {
                name: float(models[name].predict_proba(model_input)[0][1])
                for name in ENSEMBLE_MODELS
            }
            probability = sum(model_probabilities.values()) / len(model_probabilities)
    except Exception as e:
        render_error_state(
            f"Prediction failed: {str(e)}",
            "Check model compatibility with preprocessing pipeline."
        )
        st.stop()

    if probability >= 0.60 and debt_to_income <= 0.36:
        prediction = "Approved"
        decision_explanation = "The model score is at least 60% and affordability is within the lower-risk demo band."
    else:
        prediction = "Rejected"
        reasons = []
        if probability < 0.60:
            reasons.append("the five-model average is below the 60% approval cutoff")
        if debt_to_income > 0.36:
            reasons.append("payment-to-income DTI exceeds the 36% prototype limit")
        decision_explanation = "Rejected because " + " and ".join(reasons) + "."

    probability_band = (
        "Rejected score band (<60%)" if probability < 0.60 else
        "Approval score band (60%+)"
    )
    credit_label = "Good" if credit_history == 1.0 else "Poor / not established"
    dti_text = (
        f"{debt_to_income * 100:.2f}%"
        if math.isfinite(debt_to_income) else "unavailable (no household income)"
    )
    decision_basis = (
        f"Status: {prediction}. Five-model average: {probability * 100:.2f}% "
        f"({probability_band}). Estimated monthly payment: {estimated_monthly_payment:,.2f}; "
        f"existing monthly debt: {existing_monthly_debt:,.2f}; gross household monthly income: "
        f"{total_income:,.2f}; payment-to-income DTI: {dti_text} ({affordability_status}). "
        f"Credit history: {credit_label}. Property area: {property_area}. "
        f"Marital status ({married}) is recorded as context and is not scored. "
        "DTI limits and probability bands are illustrative prototype rules."
    )

    section_title("🎯 Loan Approval Result")
    result_main_col, result_gauge_col = st.columns([3, 2])
    with result_main_col:
        if prediction == "Approved":
            st.markdown(
                '<div class="alert-box success approval-status"><b>Approval status: APPROVED</b></div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="alert-box danger approval-status"><b>Approval status: REJECTED</b></div>',
                unsafe_allow_html=True
            )
        st.markdown(f"**Decision basis:** {decision_basis}")
    with result_gauge_col:
        st.metric("Five-model approval estimate", f"{probability * 100:.2f}%")
        st.progress(int(round(probability * 100)), text="Ensemble estimate")
        st.caption("60%+ is the approval score cutoff; DTI must also be 36% or less.")
        st.metric(
            "Payment-to-income ratio",
            f"{debt_to_income * 100:.2f}%" if math.isfinite(debt_to_income) else "Unavailable",
            affordability_status
        )
        st.caption(affordability_note)
    render_alert(
        "<b>Decision Support Notice</b> — This is an academic ML prototype recommendation, "
        "not a final lending decision. Always combine with manual underwriting, "
        "organisational policy checks, and regulatory compliance review.",
        kind="warning"
    )

    agreement_df = pd.DataFrame(
        [
            {
                "Model": name,
                "Approval Probability": f"{model_probabilities[name] * 100:.2f}%",
                "Individual indication (50%)": "Positive" if model_probabilities[name] >= 0.5 else "Negative",
            }
            for name in ENSEMBLE_MODELS
        ]
    )
    section_title("🧩 Model Contributions")
    st.dataframe(agreement_df, width="stretch", hide_index=True)

    section_title("🧾 Key Financial & Decision KPIs")
    det1, det2, det3, det4, det5 = st.columns(5)

    with det1:
        st.metric(
            "Total Household Income",
            f"{total_income:,.2f}",
            help="Applicant + Co-applicant income"
        )

    with det2:
        st.metric(
            "Requested Loan Amount",
            f"{loan_principal:,.0f}",
            f"{loan_term/12:.0f}-year term; amount entered in thousands"
        )

    with det3:
        st.metric(
            "Estimated Monthly Payment",
            f"{estimated_monthly_payment:,.2f}",
            f"at {annual_interest_rate:.2f}% annual interest"
        )

    with det4:
        st.metric(
            "Payment-to-Income DTI",
            f"{debt_to_income * 100:.1f}%" if math.isfinite(debt_to_income) else "Unavailable",
            affordability_status
        )

    with det5:
        st.metric("Five-model estimate", f"{probability * 100:.1f}%", prediction)

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
                ("Recommendation", prediction),
                ("Approval Probability", f"{probability * 100:.2f}%"),
                ("Decision Explanation", decision_explanation),
                ("Applicant Income", f"{applicant_income:,.2f}"),
                ("Co-applicant Income", f"{coapplicant_income:,.2f}"),
                ("Total Income (calc.)", f"{total_income:,.2f}"),
                ("Marital Status (context)", married),
                ("Property Area (model feature)", property_area),
                ("Loan Amount (thousands)", f"{loan_amount:,.2f}"),
                ("Annual Interest Rate", f"{annual_interest_rate:.2f}%"),
                ("Existing Monthly Debt", f"{existing_monthly_debt:,.2f}"),
                ("Estimated Monthly Payment", f"{estimated_monthly_payment:,.2f}"),
                ("Monthly Payment DTI", f"{debt_to_income * 100:.2f}%" if math.isfinite(debt_to_income) else "Unavailable"),
                ("Affordability Band", affordability_status),
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
                property_area,
                marital_status,
                dependents,
                self_employed,
                loan_term_months,
                annual_interest_rate,
                existing_monthly_debt,
                estimated_monthly_payment,
                debt_to_income,
                decision_explanation,
                decision_basis
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                ensemble_name,
                prediction,
                float(probability),
                applicant_income,
                coapplicant_income,
                total_income,
                loan_amount,
                loan_to_income,
                credit_history,
                education,
                property_area,
                married,
                dependents,
                self_employed,
                loan_term,
                annual_interest_rate,
                existing_monthly_debt,
                estimated_monthly_payment,
                debt_to_income if math.isfinite(debt_to_income) else None,
                decision_explanation,
                decision_basis
            )
        )
        prediction_id = cursor.lastrowid
        cursor.execute(
            """
            INSERT INTO users
            (
                prediction_id, created_at, gender, marital_status, dependents,
                education, self_employed, applicant_income, coapplicant_income,
                total_income, credit_history, property_area, loan_amount,
                loan_term_months, estimated_monthly_payment, existing_monthly_debt,
                debt_to_income, approval_status, approval_probability, decision_basis
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                prediction_id,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                gender,
                married,
                dependents,
                education,
                self_employed,
                applicant_income,
                coapplicant_income,
                total_income,
                credit_history,
                property_area,
                loan_amount,
                loan_term,
                estimated_monthly_payment,
                existing_monthly_debt,
                debt_to_income if math.isfinite(debt_to_income) else None,
                prediction,
                float(probability),
                decision_basis
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

st.markdown("---")
section_title("🧮 Financial Calculations")

financial_columns = st.columns(2)
with financial_columns[0]:
    st.metric("Applicant Gross Monthly Income", f"{applicant_income:,.2f}")
    st.metric("Co-applicant Gross Monthly Income", f"{coapplicant_income:,.2f}")
    st.metric("Total Gross Monthly Income", f"{total_income:,.2f}")
with financial_columns[1]:
    st.metric("Loan Principal", f"{loan_principal:,.2f}")
    st.metric("Loan Term", f"{loan_term} months ({loan_term / 12:.0f} years)")
    st.metric("Estimated Monthly Payment", f"{estimated_monthly_payment:,.2f}")
    st.metric("Existing Monthly Debt", f"{existing_monthly_debt:,.2f}")
    st.metric(
        "Payment-to-Income DTI",
        f"{debt_to_income * 100:.2f}%" if math.isfinite(debt_to_income) else "Unavailable",
        affordability_status
    )

with st.expander("ℹ️  How affordability is calculated", expanded=False):
    st.markdown(
        f"""
        **Monthly payment** is estimated from the principal, annual interest rate, and loan term.
        The amount field is in thousands, so the entered principal is multiplied by 1,000.

        **DTI = (estimated monthly payment + existing monthly debt) / gross monthly household income**

        - **Up to 36%:** lower-risk demonstration band.
        - **Above 36%:** does not meet the prototype's affordability requirement.

        An application is approved only when the five-model estimate is at least 60%
        and DTI is at or below 36%. Every other application is marked Rejected by this
        prototype's binary policy.
        """
    )

