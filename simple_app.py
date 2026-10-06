import streamlit as st
import pandas as pd
import joblib
import os
import sys
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import numpy as np

# ============================================================
# PROJECT PATH SETUP
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")

if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from src.preprocessing import add_features

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Loan Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 1rem;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        position: relative;
        z-index: 1;
    }
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 800;
        line-height: 1.2;
    }
    .main-header p {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
        font-size: 1.1rem;
    }
    .form-container {
        background: white;
        padding: 2rem;
        border-radius: 1rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
    }
    .result-approved {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        animation: slideIn 0.5s ease-out;
    }
    .result-rejected {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        animation: slideIn 0.5s ease-out;
    }
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 1rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        color: #667eea;
    }
    .metric-label {
        color: #666;
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 0.5rem;
        width: 100%;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(102, 126, 234, 0.4);
    }
    .info-box {
        background: #f0f4ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>🏦 Loan Approval Prediction</h1>
    <p>AI-powered loan decision support system</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_model():
    models_path = os.path.join(PROJECT_ROOT, "models")
    
    # Try to load models in order of preference (sklearn models first)
    model_files = [
        "random_forest.pkl",
        "random_forest_tuned.pkl",
        "logistic_regression.pkl",
        "decision_tree.pkl",
        "svm.pkl",
        "xgboost.pkl",
        "xgboost_tuned.pkl"
    ]
    
    for model_file in model_files:
        path = os.path.join(models_path, model_file)
        if os.path.exists(path):
            try:
                model = joblib.load(path)
                return model, model_file
            except Exception as e:
                st.warning(f"Could not load {model_file}: {str(e)}")
                continue
    
    return None, None

model, model_name = load_model()

if model is None:
    st.error("No trained models found. Please run the training pipeline first.")
    st.stop()

# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "🌳 Model Visualization", "📚 Model Information"])

# ============================================================
# TAB 1: PREDICTION
# ============================================================

with tab1:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown('<div class="form-container">', unsafe_allow_html=True)
        st.subheader("👤 Personal Information")
        
        gender = st.selectbox("Gender", ["Male", "Female"], help="Select the applicant's gender", key="gender")
        married = st.selectbox("Marital Status", ["Yes", "No"], help="Is the applicant married?", key="married")
        dependents = st.selectbox("Number of Dependents", [0, 1, 2, 3], help="Number of people dependent on the applicant", key="dependents")
        education = st.selectbox("Education Level", ["Graduate", "Not Graduate"], help="Highest education qualification", key="education")
        self_employed = st.selectbox("Self Employed", ["No", "Yes"], help="Is the applicant self-employed?", key="self_employed")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="form-container">', unsafe_allow_html=True)
        st.subheader("💰 Financial Information")
        
        applicant_income = st.number_input("Applicant Income", min_value=0, value=5000, step=100, help="Monthly/Annual income of the applicant", key="applicant_income")
        coapplicant_income = st.number_input("Coapplicant Income", min_value=0, value=0, step=100, help="Income of co-applicant (if any)", key="coapplicant_income")
        loan_amount = st.number_input("Loan Amount (in thousands)", min_value=0, value=150, step=10, help="Requested loan amount", key="loan_amount")
        loan_term = st.selectbox("Loan Term (months)", [360, 180, 120, 60], format_func=lambda x: f"{x} months ({x//12} years)", help="Repayment period", key="loan_term")
        credit_history = st.selectbox("Credit History", [1.0, 0.0], format_func=lambda x: "✅ Good" if x == 1.0 else "❌ Poor", help="Previous credit repayment record", key="credit_history")
        property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"], help="Location of the property", key="property_area")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div style="text-align: center; margin: 1rem 0;">', unsafe_allow_html=True)
    with st.form("loan_form"):
        submitted = st.form_submit_button("🔮 Predict Loan Status", type="primary")
    st.markdown('</div>', unsafe_allow_html=True)

    # ============================================================
    # PREDICTION LOGIC (inside tab1)
    # ============================================================

    if submitted:
        # Show loading state
        with st.spinner("🔍 Analyzing application..."):
            # Create input dataframe
            input_data = pd.DataFrame({
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
            })
            
            # Preprocess
            input_data = add_features(input_data)
            
            # Predict
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0][1]
        
        # Display result immediately below form (no separator)
        if prediction == 1:
            st.markdown(f"""
            <div class="result-approved">
                <h1 style="margin: 0; font-size: 3rem;">✅</h1>
                <h2 style="margin: 0.5rem 0; color: #065f46;">Loan Approved!</h2>
                <p style="margin: 0; font-size: 1.2rem; color: #064e3b;">Congratulations! Your loan application has been approved.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-rejected">
                <h1 style="margin: 0; font-size: 3rem;">❌</h1>
                <h2 style="margin: 0.5rem 0; color: #991b1b;">Loan Rejected</h2>
                <p style="margin: 0; font-size: 1.2rem; color: #7f1d1d;">Unfortunately, your loan application could not be approved at this time.</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Display metrics immediately below result
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value">{:.1f}%</div>
                <div class="metric-label">Approval Probability</div>
            </div>
            """.format(probability * 100), unsafe_allow_html=True)
        
        with col2:
            total_income = applicant_income + coapplicant_income
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value">${:,.0f}</div>
                <div class="metric-label">Total Income</div>
            </div>
            """.format(total_income), unsafe_allow_html=True)
        
        with col3:
            lti = loan_amount / (total_income + 1)
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value">{:.3f}</div>
                <div class="metric-label">Loan-to-Income Ratio</div>
            </div>
            """.format(lti), unsafe_allow_html=True)
        
        # Additional information immediately below metrics
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.info(f"🤖 **Model Used:** {model_name.replace('.pkl', '').replace('_', ' ').title()}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Key factors immediately below
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("### 📊 Key Factors Considered")
        factors = []
        if credit_history == 1.0:
            factors.append("✅ Good Credit History")
        else:
            factors.append("❌ Poor Credit History")
        
        if education == "Graduate":
            factors.append("🎓 Graduate Education")
        
        if married == "Yes":
            factors.append("💒 Married Status")
        
        if property_area == "Urban":
            factors.append("🏙️ Urban Property")
        
        st.markdown(", ".join(factors))
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Detailed explanation immediately below
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        
        if prediction == 1:
            st.markdown("### ✅ Why This Loan Was Approved")
            st.markdown("The loan was approved based on the following positive factors:")
        else:
            st.markdown("### ❌ Why This Loan Was Rejected")
            st.markdown("The loan was rejected due to the following factors:")
        
        st.markdown("---")
        
        # Credit history analysis
        if credit_history == 1.0:
            st.success("✅ **Good Credit History/Credit Score** - Applicant has no major defaults or serious repayment problems. Strong repayment history supports creditworthiness.")
        else:
            st.error("❌ **Poor Credit History or Significant Defaults** - Poor or no credit history is a major red flag. Past repayment behavior is the best predictor of future behavior.")
        
        # Income analysis
        total_income = applicant_income + coapplicant_income
        if total_income > 5000:
            st.success(f"✅ **Sufficient and Stable Income** - Salary/business income is adequate for the requested loan. Employment/business stability is favorable.")
        elif total_income > 3000:
            st.warning(f"⚠️ **Income: MODERATE (${total_income:,.0f})** - Income may be sufficient with good credit history but requires careful assessment of repayment capacity.")
        else:
            st.error(f"❌ **Insufficient Income** - Income is too low to comfortably support loan repayment. Salary/business income is not adequate for the requested loan.")
        
        # Loan-to-income ratio (repayment capacity)
        lti = loan_amount / (total_income + 1)
        if lti < 0.03:
            st.success(f"✅ **Strong Repayment Capacity** - The bank confirms applicant can comfortably handle the proposed EMI along with existing debt obligations. Loan amount is reasonable relative to income.")
        elif lti < 0.05:
            st.warning(f"⚠️ **Repayment Capacity: ACCEPTABLE ({lti:.3f})** - EMI is manageable but requires verification of existing debt burden.")
        else:
            st.error(f"❌ **Poor Repayment Capacity** - Loan requested is too large relative to repayment capacity. Higher existing debt can reduce eligibility.")
        
        # Employment/business profile
        if self_employed == "No":
            st.success("✅ **Employment Profile: STABLE** - Salaried employment provides stability and continuity of income, which is favorable for loan approval.")
        else:
            st.warning("⚠️ **Business Profile: SELF-EMPLOYED** - Self-employment income can be variable. Stability and continuity of business are considered.")
        
        # Education impact (employment stability)
        if education == "Graduate":
            st.success("✅ **Education: GRADUATE** - Higher education often correlates with stable employment and better earning potential.")
        else:
            st.info("ℹ️ **Education: NOT GRADUATE** - Education level is a neutral factor but may affect employment stability assessment.")
        
        # Property area (collateral/security)
        if property_area == "Urban":
            st.success("✅ **Collateral/Security: ACCEPTABLE** - Urban properties typically have better resale value and meet loan-to-value requirements.")
        elif property_area == "Semiurban":
            st.info("ℹ️ **Collateral/Security: MODERATE** - Semiurban properties have acceptable collateral value.")
        else:
            st.warning("⚠️ **Collateral/Security: RURAL** - Rural properties may have lower resale value and may not meet lender's collateral requirements.")
        
        # Marital status (income stability)
        if married == "Yes":
            st.success("✅ **Financial Profile: STABLE** - Married applicants often have dual income support, improving overall financial stability.")
        
        st.markdown("---")
        st.markdown("### 📊 Banking Approval Criteria")
        st.markdown("""
        **Loan is generally considered for approval when:**
        - ✅ Good credit history/credit score with no major defaults
        - ✅ Strong repayment history supports creditworthiness
        - ✅ Sufficient and stable income for the requested loan
        - ✅ Employment/business stability is favorable
        - ✅ Applicant can comfortably handle EMI with existing debt
        - ✅ Loan amount is reasonable relative to income and collateral
        - ✅ Complete and verifiable documents (KYC, income, bank statements)
        - ✅ Acceptable overall risk profile
        - ✅ Collateral/security meets lender's requirements
        
        **Loan may be rejected/declined when:**
        - ❌ Poor credit history or significant defaults
        - ❌ Insufficient income
        - ❌ High existing debt/EMI burden
        - ❌ Loan requested too large relative to repayment capacity
        - ❌ Unstable or unverifiable income
        - ❌ Missing or unverifiable documents
        - ❌ Fails required KYC/verification
        - ❌ Collateral/security doesn't meet requirements
        - ❌ Overall credit risk outside lending criteria
        
        **Note:** This AI model considers the complete financial profile rather than one factor alone, following standard banking practices.
        """)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Loan terms guidance (only if approved)
        if prediction == 1:
            st.markdown('<div class="info-box" style="background: #f0fdf4; border-left-color: #16a34a;">', unsafe_allow_html=True)
            st.markdown("### 💰 Loan Terms Guidance")
            
            # Calculate interest rate based on risk factors
            base_rate = 8.5  # Base interest rate
            
            if credit_history == 0.0:
                base_rate += 2.0
            if education != "Graduate":
                base_rate += 0.5
            if property_area == "Rural":
                base_rate += 0.5
            if lti > 0.05:
                base_rate += 1.0
            
            # Calculate monthly payment using amortization formula
            monthly_rate = base_rate / 12 / 100
            if monthly_rate > 0:
                monthly_payment = loan_amount * 1000 * (monthly_rate * (1 + monthly_rate) ** loan_term) / ((1 + monthly_rate) ** loan_term - 1)
            else:
                monthly_payment = (loan_amount * 1000) / loan_term
            
            total_payment = monthly_payment * loan_term
            total_interest = total_payment - (loan_amount * 1000)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Interest Rate", f"{base_rate:.2f}%")
            
            with col2:
                st.metric("Monthly Payment", f"${monthly_payment:,.2f}")
            
            with col3:
                st.metric("Total Payment", f"${total_payment:,.2f}")
            
            with col4:
                st.metric("Total Interest", f"${total_interest:,.2f}")
            
            st.markdown(f"""
            **Loan Summary:**
            - **Principal Amount:** ${loan_amount * 1000:,.2f}
            - **Interest Rate:** {base_rate:.2f}% per annum
            - **Loan Term:** {loan_term} months ({loan_term // 12} years)
            - **Monthly EMI:** ${monthly_payment:,.2f}
            - **Total Amount Payable:** ${total_payment:,.2f}
            
            *Note: Interest rates are estimated based on risk assessment. Actual rates may vary based on lender policies and market conditions.*
            """)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Improvement suggestions (if rejected)
        if prediction == 0:
            st.markdown('<div class="info-box" style="background: #fef2f2; border-left-color: #dc2626;">', unsafe_allow_html=True)
            st.markdown("### 💡 How to Improve Your Chances")
            st.markdown("Based on banking approval criteria, here's how to improve your application:")
            
            if credit_history == 0.0:
                st.markdown("""
                **📌 Build Good Credit History (CRITICAL)**
                - Maintain a good repayment record for 6-12 months
                - Pay all bills and EMIs on time to avoid defaults
                - Keep credit utilization below 30%
                - No major defaults or serious repayment problems
                """)
            
            if total_income < 5000:
                st.markdown(f"""
                **📌 Ensure Sufficient and Stable Income**
                - Current income (${total_income:,.0f}) is below optimal threshold
                - Add a co-applicant with higher income
                - Wait until income increases through job growth
                - Ensure employment/business stability
                """)
            
            if lti > 0.05:
                recommended_loan = total_income * 0.04 * 1000
                st.markdown(f"""
                **📌 Improve Repayment Capacity**
                - Current loan amount is too large relative to income
                - Recommended maximum: ${recommended_loan:,.0f}
                - Reduce existing debt/EMI burden
                - Ensure you can comfortably handle EMI with existing debt
                """)
            
            if education != "Graduate":
                st.markdown("""
                **📌 Education Profile (Optional)**
                - Complete education if possible
                - Professional certifications can help
                - Improves employment stability assessment
                """)
            
            if property_area == "Rural":
                st.markdown("""
                **📌 Collateral/Security**
                - Properties in urban/semiurban areas have better approval rates
                - Consider properties with better resale value
                - Ensure collateral meets lender's loan-to-value requirements
                """)
            
            if self_employed == "Yes":
                st.markdown("""
                **📌 Employment/Business Profile**
                - Demonstrate stability and continuity of business
                - Provide verifiable income documents
                - Show consistent business income over time
                """)
            
            st.markdown("---")
            st.markdown("""
            **📋 Document Requirements:**
            - Complete and verifiable KYC documents
            - Income proof (salary slips, business records)
            - Bank statements for verification
            - All documents must be genuine and verifiable
            
            **💡 Tip:** Focus on building good credit history first as it's the most critical factor. The bank considers the complete financial profile rather than one factor alone.
            """)
            
            st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# TAB 2: MODEL VISUALIZATION
# ============================================================

with tab2:
    st.markdown('<div class="form-container">', unsafe_allow_html=True)
    st.subheader("🌳 Decision Tree Visualization")
    
    if hasattr(model, 'estimators_'):
        st.info("Random Forest model detected. Showing the first decision tree from the ensemble.")
        tree_to_plot = model.estimators_[0]
    elif hasattr(model, 'tree_'):
        st.info("Decision Tree model detected.")
        tree_to_plot = model
    else:
        st.warning("The current model is not a tree-based model (e.g., Logistic Regression, SVM). Decision tree visualization is not available.")
        st.info(f"Current model: {model_name}")
        st.markdown('</div>', unsafe_allow_html=True)
        st.stop()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(20, 10))
    
    try:
        plot_tree(
            tree_to_plot,
            filled=True,
            rounded=True,
            feature_names=[
                "Gender", "Married", "Dependents", "Education", "Self_Employed",
                "ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term",
                "Credit_History", "Property_Area", "TotalIncome", "LoanAmountPerIncome"
            ],
            class_names=["Rejected", "Approved"],
            max_depth=3,
            fontsize=10,
            ax=ax
        )
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        st.markdown("---")
        st.markdown("### 📖 How to Read the Decision Tree")
        st.markdown("""
        - **Root Node:** The top node represents the first decision based on the most important feature
        - **Internal Nodes:** Each node shows the feature used for splitting, threshold, and samples
        - **Leaf Nodes:** The bottom nodes show the final prediction (Approved/Rejected)
        - **Colors:** Orange indicates higher probability of rejection, blue indicates higher probability of approval
        - **gini:** Measures impurity - lower values mean more pure splits
        """)
    except Exception as e:
        st.error(f"Error visualizing tree: {str(e)}")
    
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# TAB 3: MODEL INFORMATION
# ============================================================

with tab3:
    st.markdown('<div class="form-container">', unsafe_allow_html=True)
    st.subheader("📚 Model Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🤖 Model Details")
        st.markdown(f"""
        - **Model Name:** {model_name.replace('.pkl', '').replace('_', ' ').title()}
        - **Model Type:** {type(model).__name__}
        """)
        
        if hasattr(model, 'n_estimators'):
            st.markdown(f"- **Number of Estimators:** {model.n_estimators}")
        
        if hasattr(model, 'max_depth'):
            st.markdown(f"- **Max Depth:** {model.max_depth}")
        
        if hasattr(model, 'min_samples_split'):
            st.markdown(f"- **Min Samples Split:** {model.min_samples_split}")
        
        if hasattr(model, 'min_samples_leaf'):
            st.markdown(f"- **Min Samples Leaf:** {model.min_samples_leaf}")
    
    with col2:
        st.markdown("### 📊 How the Model Works")
        
        model_type = type(model).__name__
        
        if "RandomForest" in model_type or "DecisionTree" in model_type:
            st.markdown("""
            **Decision Tree/Random Forest:**
            - The model makes decisions by asking a series of yes/no questions about the applicant's features
            - Each question splits the data into smaller groups based on the most informative features
            - Random Forest uses multiple trees and combines their predictions for better accuracy
            - Key features like Credit History, Income, and Loan Amount are typically the most important decision points
            """)
        elif "LogisticRegression" in model_type:
            st.markdown("""
            **Logistic Regression:**
            - The model calculates a weighted sum of all features
            - Each feature has a coefficient that indicates its importance and direction
            - The result is passed through a sigmoid function to get a probability between 0 and 1
            - Features with positive coefficients increase approval chances, negative coefficients decrease them
            """)
        elif "SVM" in model_type:
            st.markdown("""
            **Support Vector Machine:**
            - The model finds the best boundary (hyperplane) that separates approved and rejected applications
            - It uses kernel functions to handle non-linear relationships
            - Support vectors are the critical data points that define the decision boundary
            """)
        else:
            st.markdown(f"""
            **{model_type}:**
            - This model uses advanced machine learning techniques to predict loan approval
            - It learns patterns from historical loan data to make predictions on new applications
            """)
    
    st.markdown("---")
    
    st.markdown("### 🔑 Key Features Used")
    st.markdown("""
    The model considers the following features when making predictions:
    
    1. **Credit History** - Most important factor showing past repayment behavior
    2. **Applicant Income** - Primary source of loan repayment
    3. **Coapplicant Income** - Additional income support
    4. **Loan Amount** - Size of the requested loan
    5. **Loan Term** - Duration of the loan repayment period
    6. **Property Area** - Location of the property (Urban/Semiurban/Rural)
    7. **Education** - Educational qualification of the applicant
    8. **Marital Status** - Whether the applicant is married
    9. **Dependents** - Number of people dependent on the applicant
    10. **Self Employed** - Employment status
    11. **Gender** - Gender of the applicant
    
    **Engineered Features:**
    - **Total Income** - Sum of applicant and coapplicant income
    - **Loan Amount per Income** - Ratio of loan amount to total income
    """)
    
    st.markdown("---")
    
    st.markdown("### ⚠️ Model Limitations")
    st.markdown("""
    - This is a machine learning model trained on historical data
    - Predictions are probabilistic, not deterministic
    - The model may not capture all real-world factors considered by human loan officers
    - Bias in training data can lead to biased predictions
    - Regular retraining with new data is recommended for optimal performance
    - This should be used as a decision support tool, not the sole decision maker
    """)
    
    st.markdown('</div>', unsafe_allow_html=True)
