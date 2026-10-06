import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sqlite3
import os
import sys
import re
import json

from datetime import datetime
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
    render_error_state,
    render_empty_state,
    render_alert,
    pro_card,
    render_page_hero,
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Loan Assistant · Loan Approval DSS",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

inject_theme_css()
render_sidebar_brand(
    app_name="Loan Assistant",
    subtitle="Interactive Help & Glossary"
)


# ============================================================
# FILE PATHS
# ============================================================

DB_PATH = os.path.join(
    PROJECT_ROOT,
    "loan_predictions.db"
)

MODELS_DIR = os.path.join(
    PROJECT_ROOT,
    "models"
)

DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "loan_data.csv"
)

IMPORTANCE_CSV = os.path.join(
    PROJECT_ROOT,
    "data",
    "shap_results",
    "shap_feature_importance.csv"
)

EVAL_CSV = os.path.join(
    PROJECT_ROOT,
    "evaluation_results.csv"
)

TUNED_EVAL_CSV = os.path.join(
    PROJECT_ROOT,
    "tuned_model_results.csv"
)

CV_CSV = os.path.join(
    PROJECT_ROOT,
    "model_results.csv"
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("### Answer tools")
    if st.button(
        "Clear answer",
        width='stretch'
    ):
        st.session_state["loan_assistant_answer"] = None



# ============================================================
# PAGE HERO
# ============================================================

render_page_hero(
    chip_label="LOAN ASSISTANT CHAT",
    title="Loan Assistant / Chat",
    subtitle=(
        "An always-available help companion. It can interpret your latest "
        "prediction, define loan terms, explain each ML model, break down "
        "SHAP and metrics, and describe the dataset + features — all from "
        "the project's persisted files (no LLM API key required)."
    ),
    mini_stats=[
        {"label": "KB Topics", "value": "30+ Entries", "sub": "Terms · Models · SHAP", "variant": "primary"},
        {"label": "Response", "value": "Instant", "sub": "Rule-based engine", "variant": "info"},
        {"label": "Capabilities", "value": "7 Areas", "sub": "FAQ + last-prediction", "variant": "success"},
    ]
)


# ============================================================
# KNOWLEDGE BASE (deterministic FAQ intents)
# ============================================================

LOAN_TERMS = {
    "credit history": {
        "question": "What is Credit History?",
        "answer": (
            "**Credit History (0.0 / 1.0)** records whether the applicant has "
            "previously repaid debts on time and meets the underwriting credit "
            "guidelines.\n\n"
            "  - **1.0 (Good):** Applicant satisfies credit criteria. "
            "This is usually the STRONGEST single predictor of approval.\n"
            "  - **0.0 (Poor/Absent):** Missing or bad credit record. "
            "Very strong negative signal.\n"
            "If you see a very low approval probability, credit history is the "
            "first place to look."
        )
    },
    "loan to income": {
        "question": "What is Loan-to-Income (LTI)?",
        "answer": (
            "**Loan-to-Income Ratio (LTI) = Loan Amount / Total Household Income**\n\n"
            "It measures how large the requested loan is compared to the money "
            "coming in each period.\n\n"
            "  - **< 0.02 (LOW):** Comfortable affordability.\n"
            "  - **0.02 – 0.04 (MEDIUM):** Typical acceptable band.\n"
            "  - **> 0.04 (HIGH):** Risky; repayment burden looks heavy.\n\n"
            "High LTI almost always pushes approval probability DOWN."
        )
    },
    "ltv": {
        "question": "What is LTV / loan-to-value?",
        "answer": (
            "LTV = Loan Amount / Property Value. It isn't directly stored in "
            "this dataset (we don't record property purchase value, only the "
            "Property Area type: Rural / Semiurban / Urban).\n\n"
            "A close proxy here is **Loan-to-Income** (LTI) which captures "
            "affordability — and Property Area, which captures the local "
            "market risk of the collateral."
        )
    },
    "loan term": {
        "question": "What is Loan Amount Term?",
        "answer": (
            "**Loan Amount Term** is the agreed repayment period in months. "
            "Typical values are 360 (30-yr), 300 (25-yr), 240 (20-yr), 180 "
            "(15-yr), 120 (10-yr), 84 (7-yr), 60 (5-yr) and 36 (3-yr).\n\n"
            "Shorter terms usually mean higher monthly repayments but lower "
            "total interest; longer terms reduce monthly burden but cost more "
            "over the full life of the loan."
        )
    },
    "property area": {
        "question": "What does Property Area mean?",
        "answer": (
            "Categorises the collateral location into three tiers:\n\n"
            "  - **Rural** — smaller, remote property markets; higher "
            "perceived collateral risk.\n"
            "  - **Semiurban** — typically the highest approval rate band in "
            "this dataset.\n"
            "  - **Urban** — city markets, good liquidity but usually "
            "higher loan amounts requested.\n\n"
            "Area-type encodes market risk and the ease / cost of selling "
            "the collateral in the event of default."
        )
    },
    "coapplicant": {
        "question": "What is a Coapplicant?",
        "answer": (
            "A **co-applicant** (co-signer / co-borrower / guarantor) is a "
            "second person whose income and credit are combined with the "
            "primary applicant to strengthen repayment capacity.\n\n"
            "A co-applicant is beneficial whenever their income is positive, "
            "because Total Household Income rises and Loan-to-Income falls."
        )
    },
    "self employed": {
        "question": "What does Self-Employed mean?",
        "answer": (
            "**Self_Employed = Yes** indicates the applicant earns income from "
            "their own business / freelance work rather than a regular salary.\n\n"
            "Self-employment can sometimes be seen as less predictable than "
            "payroll income, but the model's opinion of it ultimately depends "
            "on how Self_Employed applicants actually performed in the "
            "training data."
        )
    },
    "dependents": {
        "question": "What are Dependents?",
        "answer": (
            "**Dependents** is the count of people (spouse, children, etc.) "
            "financially reliant on the applicant. The dataset stores this "
            "as 0, 1, 2, or 3+.\n\n"
            "More dependents usually increases the household expenditure, "
            "which can reduce available surplus for loan repayments."
        )
    },
    "probability": {
        "question": "What does Approval Probability mean?",
        "answer": (
            "**Approval Probability** is the model's confidence, from 0% to "
            "100%, that the application belongs to the *Approved* class.\n\n"
            "  - **> 50%:** predicted APPROVED\n"
            "  - **< 50%:** predicted REJECTED\n\n"
            "The 50% line is the decision threshold. 70% means the model is "
            "more confident than 50%; 20% means it strongly expects rejection.\n\n"
            "Probability is NOT a promise. It reflects only what the model "
            "learned from historical training data."
        )
    }
}


ML_MODELS = {
    "logistic regression": {
        "name": "Logistic Regression",
        "answer": (
            "**Logistic Regression** is the baseline linear classifier. It "
            "fits a logistic sigmoid curve to a weighted sum of inputs.\n\n"
            "  - Pros: very fast, highly interpretable coefficients, "
            "small chance of overfitting.\n"
            "  - Cons: only captures linear relationships, can underperform "
            "if features interact heavily.\n\n"
            "Often used as a benchmark to beat with more sophisticated "
            "ensembles."
        )
    },
    "decision tree": {
        "name": "Decision Tree",
        "answer": (
            "**Decision Tree** learns a nested sequence of if/else splits "
            "on single features (e.g. \"Credit_History ≤ 0.5? → Reject\").\n\n"
            "  - Pros: extremely interpretable, handles categorical + numeric "
            "features naturally, robust to monotonic transformations.\n"
            "  - Cons: very prone to overfit noisy data; unstable "
            "(small data changes → very different tree).\n\n"
            "Tree ensembles (Random Forest / XGBoost) fix the instability by "
            "averaging hundreds of trees."
        )
    },
    "random forest": {
        "name": "Random Forest",
        "answer": (
            "**Random Forest** trains hundreds of independent decision trees "
            "on bootstrapped subsets of rows and random subsets of features "
            "(\"bagging + feature randomness\"). Predictions are averaged.\n\n"
            "  - Pros: very high accuracy, built-in feature importance, "
            "robust to overfitting, needs very little tuning out-of-the-box.\n"
            "  - Cons: slower than a single tree, harder to interpret "
            "point-by-point (use SHAP!).\n\n"
            "The tuned Random Forest is usually a top-2 contender on this "
            "loan dataset."
        )
    },
    "svm": {
        "name": "SVM (Support Vector Machine)",
        "answer": (
            "**SVM** looks for the hyperplane that maximises the margin "
            "(\"street width\") between Approved / Rejected classes, using "
            "kernel tricks (e.g. RBF) to handle non-linear boundaries.\n\n"
            "  - Pros: excellent on mid-sized tabular data, stable, "
            "works well with feature scaling.\n"
            "  - Cons: slow on big datasets, less interpretable than trees, "
            "sensitive to the choice of C / gamma.\n\n"
            "Tuning C and gamma gives the Tuned SVM a nice lift."
        )
    },
    "xgboost": {
        "name": "XGBoost (Extreme Gradient Boosting)",
        "answer": (
            "**XGBoost** is a gradient-boosted-tree algorithm. It builds "
            "trees one-by-one, each new tree trying to correct the errors "
            "of the previous ensemble (\"boosting\").\n\n"
            "  - Pros: state-of-the-art tabular accuracy, native handling "
            "of missing values, regularisation (L1 + L2) built in.\n"
            "  - Cons: easy to overfit if hyperparameters (max_depth, "
            "learning_rate, n_estimators) are not tuned carefully.\n\n"
            "XGBoost Tuned is very often the best or second-best model on "
            "this dataset by ROC-AUC and F1."
        )
    },
    "tuning": {
        "name": "Hyperparameter Tuning",
        "answer": (
            "**Hyperparameter tuning** (RandomizedSearchCV here) tries many "
            "different model settings on a 5-fold CV of the training set "
            "and picks the one with the best ROC-AUC.\n\n"
            "  - Random Forest Tuned: n_estimators, max_depth, "
            "min_samples_split/leaf, max_features.\n"
            "  - SVM Tuned: C, gamma, kernel.\n"
            "  - XGBoost Tuned: n_estimators, max_depth, learning_rate, "
            "subsample, colsample_bytree.\n\n"
            "You can compare Base vs Tuned gains on the **📊 Model Comparison** "
            "page."
        )
    },
    "cross validation": {
        "name": "Cross-Validation",
        "answer": (
            "**5-fold Stratified Cross-Validation** splits the training set "
            "into 5 balanced parts. The model is trained on 4 parts and "
            "tested on the 5th, rotating through all 5 splits.\n\n"
            "The result is **CV Mean Accuracy** (average score across the 5 "
            "folds) and **CV Std** (how stable the score was). A low CV Std "
            "indicates a more reliable / generalisable model."
        )
    }
}


SHAP_INFO = {
    "shap": {
        "question": "What is SHAP?",
        "answer": (
            "**SHAP (SHapley Additive exPlanations)** is a game-theory method "
            "that distributes the model's output fairly across every input "
            "feature. It answers:\n\n"
            "  > *\"How much did THIS feature nudge the prediction away from "
            "the baseline average (50%)?\"*\n\n"
            "Every SHAP value has:\n"
            "  - **Sign**: + pushed towards approval, − pushed towards "
            "rejection.\n"
            "  - **Magnitude**: how big that push was.\n\n"
            "Average |SHAP| across all samples gives you the **global "
            "feature importance ranking**."
        )
    },
    "shap summary": {
        "question": "How do I read the SHAP summary plot?",
        "answer": (
            "The **SHAP beeswarm / summary plot** (🔍 Explainable AI page) "
            "shows:\n\n"
            "  - **Y axis:** Features sorted by importance (top = most "
            "influential).\n"
            "  - **X axis:** SHAP value. Positive = approval push, "
            "negative = rejection push.\n"
            "  - **Point colour:** Feature value (blue = low, red = high).\n\n"
            "If you see the top feature's red dots mostly on the right and "
            "blue dots mostly on the left, that's a clear monotonic "
            "relationship (high value → approve)."
        )
    },
    "feature importance": {
        "question": "What is Global Feature Importance?",
        "answer": (
            "Global importance is **mean(|SHAP value|)** of each feature, "
            "averaged across every sample in the holdout set.\n\n"
            "It tells you which features the model leans on most heavily "
            "OVERALL, not for any specific applicant. The ranking (top 1, "
            "top 3 share, etc.) is shown on the 🔍 Explainable AI → Global "
            "Feature Importance page."
        )
    }
}


METRICS_INFO = {
    "accuracy": {
        "question": "What is Accuracy?",
        "answer": (
            "**Accuracy = (TP + TN) / All samples**\n\n"
            "Percentage of applications classified correctly (Approved that "
            "were really Approved + Rejected that were really Rejected).\n\n"
            "⚠️ Use with caution on imbalanced classes. If 69% of all loans "
            "are Approved, a \"predict always Approved\" baseline would "
            "still get 69% accuracy. Use Precision / Recall / F1 / ROC-AUC "
            "alongside it."
        )
    },
    "precision": {
        "question": "What is Precision?",
        "answer": (
            "**Precision = TP / (TP + FP)**\n\n"
            "Of the applications the model APPROVED, how many were *truly* "
            "good risks?\n\n"
            "High precision → very few **false approvals** (Type I error: "
            "approve a bad applicant who then defaults). This matters when "
            "bad approvals are very expensive."
        )
    },
    "recall": {
        "question": "What is Recall / Sensitivity?",
        "answer": (
            "**Recall = TP / (TP + FN)**\n\n"
            "Of the applications that truly deserved to be Approved, how "
            "many did the model actually catch and approve?\n\n"
            "High recall → very few **false rejections** (Type II error: "
            "reject a good applicant, lose the business + customer). "
            "Trade-off: maximising recall usually lowers precision."
        )
    },
    "f1 score": {
        "question": "What is F1 Score?",
        "answer": (
            "**F1 = 2 · (Precision · Recall) / (Precision + Recall)**\n\n"
            "Harmonic mean of precision and recall. It's a single number "
            "that balances false approvals vs false rejections.\n\n"
            "Use F1 when you don't have a strong reason to weight one "
            "error type as much more costly than the other."
        )
    },
    "roc auc": {
        "question": "What is ROC-AUC?",
        "answer": (
            "**ROC-AUC** = Area Under the Receiver Operating Characteristic "
            "curve. ROC-AUC ranges 0.0 – 1.0, where:\n\n"
            "  - **1.0** = perfect separation: the model ranks every "
            "truly-approved application above every truly-rejected one.\n"
            "  - **0.5** = as good as random.\n"
            "  - **< 0.5** = worse than random (invert predictions!).\n\n"
            "Unlike Accuracy / F1, ROC-AUC uses the *probabilities* not "
            "only the hard 0/1 predictions, so it's good for ranking. This "
            "is the metric used for hyperparameter tuning in this project."
        )
    },
    "confusion matrix": {
        "question": "What is a Confusion Matrix?",
        "answer": (
            "A **confusion matrix** is a 2×2 grid of:\n\n"
            "  - **TN (True Negative):** correctly rejected.\n"
            "  - **FP (False Positive):** predicted Approved, actually "
            "Rejected (Type I error).\n"
            "  - **FN (False Negative):** predicted Rejected, actually "
            "Approved (Type II error).\n"
            "  - **TP (True Positive):** correctly approved.\n\n"
            "Accuracy, Precision, Recall and F1 are all derived from these "
            "4 numbers."
        )
    }
}


DATASET_INFO = {
    "dataset": {
        "question": "Tell me about the dataset.",
        "answer": (
            "**loan_data.csv** contains ~614 rows × 13 columns of historical "
            "loan applications.\n\n"
            "Target column: `Loan_Status` (Y = Approved, N = Rejected). "
            "Class balance is roughly 68.7% Approved, 31.3% Rejected.\n\n"
            "Rows contain demographics, income, loan amount, term, credit "
            "history and property area. Small fractions of cells are "
            "missing (see 📈 Analytics → Missing Value Statistics)."
        )
    },
    "features": {
        "question": "Explain the features / columns.",
        "answer": (
            "The 12 input features are:\n\n"
            "**Demographics** – Loan_ID (ID, dropped from training), Gender, "
            "Married, Dependents, Education, Self_Employed.\n\n"
            "**Financial** – ApplicantIncome, CoapplicantIncome, LoanAmount, "
            "Loan_Amount_Term.\n\n"
            "**Underwriting signals** – Credit_History (0.0 or 1.0) + "
            "Property_Area (Rural/Semiurban/Urban).\n\n"
            "**Engineered features (during preprocessing)** – Total_Income "
            "= Applicant + Coapplicant; Loan_to_Income = LoanAmount / "
            "Total_Income. These two typically sit in the top 5 of the "
            "global SHAP ranking."
        )
    },
    "missing values": {
        "question": "How are missing values handled?",
        "answer": (
            "The training preprocessor (`ColumnTransformer` in the "
            "pipeline) uses:\n\n"
            "  - **SimpleImputer (mean)** for numeric columns.\n"
            "  - **SimpleImputer (most_frequent)** for categorical columns.\n\n"
            "On the Analytics page you can inspect the exact share of "
            "missing cells per column and whether missingness correlates "
            "with lower/higher approval rates."
        )
    }
}


# ============================================================
# INTENT HELPER
# ============================================================

def _matches(text: str, keywords) -> bool:
    t = (text or "").lower()
    return any(k.lower() in t for k in keywords)


def classify_intent(text: str) -> str:
    """Heuristic classifier of user intents. Returns an intent string."""

    t = (text or "").strip().lower()

    if not t:
        return "empty"

    if _matches(t, ["explain prediction", "explain my prediction", "my prediction",
                    "my application", "last prediction", "latest prediction",
                    "why approve", "why reject", "why did i", "why did the model",
                    "explain the prediction"]):
        return "explain_prediction"

    if _matches(t, ["probability", "confidence", "50%", "threshold"]):
        if _matches(t, ["my", "prediction", "application"]):
            return "explain_prediction"
        return "loan_terms::probability"

    if _matches(t, ["loan terms", "term", "lti", "loan to income",
                    "ltv", "loan-to-value", "credit history",
                    "coapplicant", "co-applicant", "co applicant",
                    "property area", "self-employed", "self employed",
                    "dependent", "marital", "married", "gender",
                    "education"]):
        if _matches(t, ["credit"]): return "terms::credit history"
        if _matches(t, ["ltv", "loan-to-value", "loan to value"]): return "terms::ltv"
        if _matches(t, ["lti", "loan to income", "loan-to-income"]): return "terms::loan to income"
        if _matches(t, ["term"]): return "terms::loan term"
        if _matches(t, ["property"]): return "terms::property area"
        if _matches(t, ["coapplicant", "co-applicant", "co applicant"]): return "terms::coapplicant"
        if _matches(t, ["self-employed", "self employed"]): return "terms::self employed"
        if _matches(t, ["dependent"]): return "terms::dependents"
        return "terms::list"

    if _matches(t, ["model", "models", "logistic", "decision tree", "random forest",
                    "svm", "xgboost", "classifier", "tuning", "tuned",
                    "cross validation", "cv"]):
        if _matches(t, ["logistic"]): return "model::logistic regression"
        if _matches(t, ["tree"]): return "model::decision tree"
        if _matches(t, ["forest"]): return "model::random forest"
        if _matches(t, ["svm", "support vector"]): return "model::svm"
        if _matches(t, ["xgb", "gradient boost", "xgboost"]): return "model::xgboost"
        if _matches(t, ["tuning", "tuned", "hyperparameter"]): return "model::tuning"
        if _matches(t, ["cross validation", "cv", "fold"]): return "model::cross validation"
        return "model::list"

    if _matches(t, ["shap", "summary plot", "beeswarm", "feature importance",
                    "explainable", "interpretability", "contribution"]):
        if _matches(t, ["summary"]): return "shap::summary"
        if _matches(t, ["importance", "global"]): return "shap::importance"
        return "shap::shap"

    if _matches(t, ["metric", "metrics", "accuracy", "precision", "recall",
                    "f1", "roc", "auc", "confusion"]):
        if _matches(t, ["accuracy"]): return "metric::accuracy"
        if _matches(t, ["precision"]): return "metric::precision"
        if _matches(t, ["recall", "sensitivity"]): return "metric::recall"
        if _matches(t, ["f1", "f-score", "f score"]): return "metric::f1 score"
        if _matches(t, ["roc", "auc", "roc-auc"]): return "metric::roc auc"
        if _matches(t, ["confusion"]): return "metric::confusion matrix"
        return "metric::list"

    if _matches(t, ["dataset", "data source", "data set", "columns", "feature",
                    "input", "missing value", "nan", "null"]):
        if _matches(t, ["missing", "nan", "null"]): return "data::missing values"
        if _matches(t, ["column", "feature", "input"]): return "data::features"
        return "data::dataset"

    if _matches(t, ["help", "hi", "hello", "hey", "what can you do",
                    "start over", "help me"]):
        return "help"

    return "unknown"


# ============================================================
# DATA LOADERS
# ============================================================

@st.cache_data
def get_prediction_history():
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    conn = sqlite3.connect(DB_PATH)
    try:
        return pd.read_sql_query(
            "SELECT * FROM predictions ORDER BY id DESC",
            conn
        )
    finally:
        conn.close()


@st.cache_data
def load_metrics_df():
    frames = []
    if os.path.exists(EVAL_CSV):
        d = pd.read_csv(EVAL_CSV); d["Variant"] = "Base"; frames.append(d)
    if os.path.exists(TUNED_EVAL_CSV):
        d = pd.read_csv(TUNED_EVAL_CSV); d["Variant"] = "Tuned"; frames.append(d)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True, sort=False)


@st.cache_data
def load_shap_importance():
    if not os.path.exists(IMPORTANCE_CSV):
        return None
    return pd.read_csv(IMPORTANCE_CSV)


@st.cache_data
def load_raw_data():
    if not os.path.exists(DATA_PATH):
        return None
    return pd.read_csv(DATA_PATH)


# ============================================================
# RESPONSE GENERATORS
# ============================================================

def render_list(title, items):
    out = [f"**{title}:**", ""]
    for label, short in items:
        out.append(f"  • **{label}** — {short}")
        return "\n".join(out) + "\n\nSelect a suggested question to see a detailed answer."


def generate_response(user_text: str) -> str:

    intent = classify_intent(user_text)
    hist = get_prediction_history()
    metrics = load_metrics_df()
    shap_df = load_shap_importance()
    data_df = load_raw_data()

    # -------- HELP --------
    if intent == "empty":
        return "I'm ready! Start with one of the suggestion chips above the chat input 😊"

    if intent == "help":
        return (
            "Hi! I'm your 🏦 **Loan Assistance bot**. I can:\n\n"
            "  1. **Explain your prediction** – try \"explain my last prediction\"\n"
            "  2. **Define loan terms** – credit history, LTI, property area, etc.\n"
            "  3. **Explain ML models** – Logistic, Tree, RF, SVM, XGB, Tuned, CV\n"
            "  4. **Explain SHAP / feature importance / summary plot**\n"
            "  5. **Explain metrics** – Accuracy/Precision/Recall/F1/ROC-AUC/Confusion\n"
            "  6. **Talk about the dataset & features & missing-value handling**\n"
            "  7. **Interpret probability/threshold meaning**\n\n"
        "Select a suggested question below to explore topics such as:\n"
            "  • What is ROC-AUC?\n"
            "  • Explain Random Forest\n"
            "  • Explain my last prediction\n"
            "  • What's credit history?\n"
        )

    # -------- EXPLAIN LATEST PREDICTION --------
    if intent == "explain_prediction":
        if hist is None or hist.empty:
            return (
                "I don't see any predictions in your local Prediction "
                "History yet. 😕\n\nPlease run at least one prediction "
                "from the **🔮 Prediction** page (or **🤝 Model Agreement**) "
                "and come back here. Alternatively, ask me general questions "
                "like \"What is Credit History?\" or \"Explain XGBoost\"."
            )

        last = hist.iloc[0]
        p = float(last.get("probability", 0.0))
        outcome = last.get("prediction", "?")
        model = last.get("model", "Unknown")
        ts = last.get("timestamp", "?")
        credit = last.get("credit_history", np.nan)
        income = last.get("applicant_income", 0.0)
        co_inc = last.get("coapplicant_income", 0.0)
        total_inc = last.get("total_income", 0.0) or (income + co_inc)
        loan = last.get("loan_amount", 0.0)
        lti = last.get("loan_to_income", None) or (loan / (total_inc + 1))
        edu = last.get("education", "N/A")
        area = last.get("property_area", "N/A")

        risk = "LOW 🟢"
        if 0.02 <= lti < 0.04: risk = "MEDIUM 🟡"
        elif lti >= 0.04: risk = "HIGH 🔴"

        # Build top drivers
        reasons_for = []
        reasons_against = []

        if credit == 1.0:
            reasons_for.append(("💳 Good Credit History",
                                "Biggest single approving driver."))
        else:
            reasons_against.append(("💳 Poor / Absent Credit History",
                                    "Largest single rejecting driver."))

        if lti < 0.02:
            reasons_for.append(("📐 Low LTI",
                                f"LTI = {lti:.4f} indicates comfortable affordability."))
        elif lti >= 0.04:
            reasons_against.append(("📐 Elevated LTI",
                                    f"LTI = {lti:.4f} flags a heavy repayment burden."))

        if edu == "Graduate":
            reasons_for.append(("🎓 Graduate education",
                                "Correlated with higher repayment capacity."))
        else:
            reasons_against.append(("📚 Not Graduate",
                                    "Slight downward pressure on approval probability."))

        if area == "Semiurban":
            reasons_for.append(("🏘️ Semiurban property area",
                                "This area usually has the highest approval rate band."))
        elif area == "Rural":
            reasons_against.append(("🌾 Rural property area",
                                    "Collateral market risk may be priced in lower."))

        if co_inc and float(co_inc) > 0:
            reasons_for.append(("🤝 Positive Co-applicant Income",
                                f"Adds {float(co_inc):,.2f} to household income."))

        total_income_ok = total_inc >= 5000
        if total_income_ok:
            reasons_for.append(("👥 Above-median Total Income",
                                f"{total_inc:,.2f} strengthens repayment capacity."))
        else:
            reasons_against.append(("👥 Low Total Household Income",
                                    f"{total_inc:,.2f} is below dataset median."))

        verdict_sentence = (
            f"Prediction #{last['id']} ({ts}) with **{model}** returned "
            f"**{outcome.upper()}** with an approval probability of "
            f"**{p*100:.2f}%** (decision threshold is 50%).\n\n"
        )

        profile = (
            "**Profile snapshot:**\n"
            f"  • Credit History: **{'Good (1.0)' if credit==1.0 else 'Poor (0.0)'}"
            f"**\n  • Total Income: **{total_inc:,.2f}** (Applicant {income:,.2f}"
            f" + Co-applicant {co_inc:,.2f})\n"
            f"  • Loan Amount: **{loan:,.2f}**, Loan-to-Income = **{lti:.4f}"
            f" ({risk})**\n"
            f"  • Education: **{edu}**, Property Area: **{area}**\n\n"
        )

        drivers = "**Why the model made this prediction?**\n"
        if reasons_for:
            drivers += "\n**Approving (positive) drivers:**\n"
            for label, detail in reasons_for:
                drivers += f"  ✅ {label} — {detail}\n"
        if reasons_against:
            drivers += "\n**Rejecting (negative) drivers:**\n"
            for label, detail in reasons_against:
                drivers += f"  ❌ {label} — {detail}\n"

        if shap_df is not None and len(shap_df):
            top2 = shap_df.head(2)
            drivers += (
                "\n💡 **Globally,** the 2 most influential features overall "
                "are **" + top2.iloc[0]["Feature"] + "** (weight "
                f"{top2.iloc[0]['SHAP Importance']:.4f}) and **"
                + top2.iloc[1]["Feature"] + f"** (weight "
                f"{top2.iloc[1]['SHAP Importance']:.4f})."
            )

        return verdict_sentence + profile + drivers

    # -------- LOAN TERMS --------
    if intent.startswith("terms::"):
        key = intent.split("::", 1)[1]
        if key in LOAN_TERMS:
            entry = LOAN_TERMS[key]
            return entry["answer"]
        if key == "list":
            return render_list(
                "Loan-related terms I can explain",
                [(v["question"].replace("What is ", "").replace("?", "").strip(),
                  "Type a sentence with that word to see its definition.")
                 for v in LOAN_TERMS.values()]
            )

    if intent.startswith("loan_terms::"):
        key = intent.split("::", 1)[1]
        if key in LOAN_TERMS:
            return LOAN_TERMS[key]["answer"]

    # -------- MODELS --------
    if intent.startswith("model::"):
        key = intent.split("::", 1)[1]
        if key in ML_MODELS:
            entry = ML_MODELS[key]
            suffix = ""
            if metrics is not None and entry["name"] in metrics["Model"].values:
                row = metrics[metrics["Model"] == entry["name"]].iloc[0]
                suffix = (
                    f"\n\n📊 Recorded performance:\n"
                    f"  • Accuracy = **{row['Accuracy']*100:.2f}%**\n"
                    f"  • Precision = **{row['Precision']*100:.2f}%**\n"
                    f"  • Recall = **{row['Recall']*100:.2f}%**\n"
                    f"  • F1 = **{row['F1 Score']*100:.2f}%**\n"
                    f"  • ROC-AUC = **{row['ROC-AUC']*100:.2f}%**\n"
                    f"  • Variant = **{row.get('Variant','?')}**"
                )
            return f"{entry['answer']}{suffix}"
        if key == "list":
            items = [
                (v["name"], "How it works, pros/cons + stored performance metrics.")
                for v in ML_MODELS.values() if v.get("name")
            ]
            return render_list("Machine Learning models I can explain", items)

    # -------- SHAP --------
    if intent.startswith("shap::"):
        key = intent.split("::", 1)[1]
        if key in SHAP_INFO:
            entry = SHAP_INFO[key]
            suffix = ""
            if key == "shap" and shap_df is not None:
                top = shap_df.head(5)
                rows = "\n".join(
                    f"  {i+1}. **{r['Feature']}** — mean|SHAP| = "
                    f"{r['SHAP Importance']:.4f}"
                    for i, r in top.iterrows()
                )
                suffix = (
                    f"\n\n🔍 **Top 5 global features on the actual saved SHAP CSV:**\n{rows}"
                )
            return f"{entry['answer']}{suffix}"

    # -------- METRICS --------
    if intent.startswith("metric::"):
        key = intent.split("::", 1)[1]
        if key in METRICS_INFO:
            entry = METRICS_INFO[key]
            suffix = ""
            if metrics is not None and key != "confusion matrix":
                col = {
                    "accuracy": "Accuracy",
                    "precision": "Precision",
                    "recall": "Recall",
                    "f1 score": "F1 Score",
                    "roc auc": "ROC-AUC"
                }.get(key)
                if col and col in metrics.columns:
                    best = metrics.iloc[metrics[col].idxmax()]
                    suffix = (
                        f"\n\n🏆 In the saved evaluation CSVs, the **best {col}** "
                        f"is **{best[col]*100:.2f}%** produced by "
                        f"**{best['Model']}** ({best.get('Variant','Base')})."
                    )
            return f"{entry['answer']}{suffix}"
        if key == "list":
            return render_list(
                "Metrics I can explain",
                [(v["question"].replace("What is ", "").replace("?", ""),
                  "Definition, formula and interpretation.")
                 for v in METRICS_INFO.values()]
            )

    # -------- DATASET / FEATURES --------
    if intent.startswith("data::"):
        key = intent.split("::", 1)[1]
        if key in DATASET_INFO:
            entry = DATASET_INFO[key]
            suffix = ""
            if key == "dataset" and data_df is not None:
                suffix = (
                    f"\n\n**Dataset (from disk):** {data_df.shape[0]:,} rows × "
                    f"{data_df.shape[1]} columns. Approval rate = "
                    f"{(data_df['Loan_Status']=='Y').mean()*100:.1f}%."
                )
            if key == "features" and shap_df is not None:
                suffix = (
                    "\n\n🔍 **Top SHAP-ranked engineered features (from saved CSV):**\n"
                    + "\n".join(
                        f"  {i+1}. **{r['Feature']}** — mean|SHAP| = {r['SHAP Importance']:.4f}"
                        for i, r in shap_df.head(5).iterrows()
                    )
                )
            return f"{entry['answer']}{suffix}"

    # -------- FALLBACKS --------

    # Try keyword-based best-effort
    all_kb_entries = (
        list(LOAN_TERMS.items()) + list(SHAP_INFO.items()) +
        list(METRICS_INFO.items()) + list(DATASET_INFO.items()) +
        list(ML_MODELS.items())
    )
    t = user_text.lower()
    matches = [(k, v) for k, v in all_kb_entries if k in t]
    if matches:
        k, v = matches[0]
        if "question" in v:
            return v["answer"]
        if "name" in v and "answer" in v:
            return v["answer"]

    return (
        "Hmm, I couldn't find a direct match. Try phrasing the question "
        "with keywords like:\n\n"
        "  • *prediction / approved / probability*\n"
        "  • *credit history / LTI / loan term / property area*\n"
        "  • *Random Forest / SVM / XGBoost / CV / tuned*\n"
        "  • *SHAP / feature importance / summary plot*\n"
        "  • *Accuracy / Precision / Recall / F1 / ROC-AUC / confusion*\n"
        "  • *dataset / features / missing values*\n"
    )


# ============================================================
# SINGLE ANSWER STATE
# ============================================================

if "loan_assistant_answer" not in st.session_state:
    st.session_state["loan_assistant_answer"] = None


st.markdown("---")

# ============================================================
# SUGGESTED QUESTIONS
# ============================================================

STATIC_SUGGESTIONS = [
    "Explain my last prediction",
    "What is Credit History?",
    "What is Loan-to-Income (LTI)?",
    "Explain Random Forest",
    "Explain XGBoost Tuned",
    "What is SHAP?",
    "What is F1 Score?",
    "What is ROC-AUC?",
    "Tell me about the dataset",
    "What are the features?",
    "What is cross validation?",
    "Explain approval probability"
]

section_title("💡 Suggested questions")

chip_cols = st.columns(3)
for i, q in enumerate(STATIC_SUGGESTIONS):
    with chip_cols[i % 3]:
        if st.button(q, width='stretch', key=f"chip_{i}"):
            st.session_state["loan_assistant_answer"] = generate_response(q)
            st.rerun()

st.markdown("---")

# ============================================================
# CURRENT ANSWER
# ============================================================

if st.session_state["loan_assistant_answer"]:
    section_title("🏦 Answer")
    st.markdown(st.session_state["loan_assistant_answer"])

st.markdown("---")

# ============================================================
# KNOWLEDGE BASE BROWSER (reference panel)
# ============================================================

with st.expander("📚 Browse the full knowledge base", expanded=False):

    tab_terms, tab_models, tab_shap, tab_metrics, tab_data = st.tabs(
        ["💳 Loan Terms", "🤖 ML Models", "🔍 SHAP", "📊 Metrics", "📈 Dataset"]
    )

    with tab_terms:
        for k, v in LOAN_TERMS.items():
            st.markdown(
                f"""
                <div class="kb-card">
                    <div class="kb-q">❓ {v['question']}</div>
                    <div class="kb-a">{v['answer'].replace(chr(10),'<br>')}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    with tab_models:
        for k, v in ML_MODELS.items():
            st.markdown(
                f"""
                <div class="kb-card">
                    <div class="kb-q">🤖 {v['name']}</div>
                    <div class="kb-a">{v['answer'].replace(chr(10),'<br>')}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    with tab_shap:
        for k, v in SHAP_INFO.items():
            st.markdown(
                f"""
                <div class="kb-card">
                    <div class="kb-q">🔍 {v['question']}</div>
                    <div class="kb-a">{v['answer'].replace(chr(10),'<br>')}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    with tab_metrics:
        for k, v in METRICS_INFO.items():
            st.markdown(
                f"""
                <div class="kb-card">
                    <div class="kb-q">📊 {v['question']}</div>
                    <div class="kb-a">{v['answer'].replace(chr(10),'<br>')}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    with tab_data:
        for k, v in DATASET_INFO.items():
            st.markdown(
                f"""
                <div class="kb-card">
                    <div class="kb-q">📈 {v['question']}</div>
                    <div class="kb-a">{v['answer'].replace(chr(10),'<br>')}</div>
                </div>
                """,
                unsafe_allow_html=True
            )


# ============================================================
# DISCLAIMER FOOTER
# ============================================================

