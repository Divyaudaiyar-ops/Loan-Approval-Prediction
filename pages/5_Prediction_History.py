import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import os
import sys

from datetime import datetime, timedelta


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
    page_title="Prediction History · Loan Approval DSS",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

inject_theme_css()
render_sidebar_brand(
    app_name="Prediction History",
    subtitle="Audit Log · Filter & Export"
)


# ============================================================
# FILE PATHS
# ============================================================

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
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("### History tools")
    if st.button(
        "Clear ALL prediction history",
        width='stretch'
    ):
        if "confirm_clear" not in st.session_state:
            st.session_state["confirm_clear"] = 1
        elif st.session_state["confirm_clear"] == 1:
            # Second click = confirm
            try:
                conn = sqlite3.connect(DB_PATH)
                cur = conn.cursor()
                cur.execute("DELETE FROM predictions;")
                conn.commit()
                conn.close()
                st.success("✅ All prediction history cleared from database.")
                st.session_state["confirm_clear"] = 0
            except Exception as e:
                render_error_state(f"Error clearing DB: {e}")

    if st.session_state.get("confirm_clear", 0) == 1:
        render_alert(
            "⚠️ Click the same button again to permanently delete all rows.",
            kind="warning"
        )



# ============================================================
# PAGE HERO
# ============================================================

render_page_hero(
    chip_label="PREDICTION HISTORY",
    title="Prediction History",
    subtitle=(
        "Complete, time-stamped audit log of every prediction run in this DSS. "
        "Every entry stores the date/time, all applicant fields, the model used, "
        "its verdict, and its approval probability. Search, filter, download "
        "and compute statistics below."
    ),
    mini_stats=[
        {"label": "Storage", "value": "SQLite DB", "sub": "Persistent local", "variant": "info"},
        {"label": "Export", "value": "CSV", "sub": "Filtered & full", "variant": "primary"},
        {"label": "Features", "value": "Search + Filters", "sub": "Date · Prob · Model", "variant": "success"},
    ]
)


# ============================================================
# LOAD HISTORY
# ============================================================

@st.cache_data(ttl=5)
def load_history():
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query(
            """
            SELECT * FROM predictions ORDER BY id DESC
            """,
            conn
        )
    finally:
        conn.close()

    if not df.empty and "timestamp" in df.columns:
        df["timestamp_dt"] = pd.to_datetime(
            df["timestamp"],
            errors="coerce"
        )
    return df


with loading_spinner("Loading prediction history from database ..."):
    history = load_history()

if history.empty:

    render_empty_state(
        "No predictions have been recorded yet.",
        (
            "Navigate to **🔮 Prediction** in the sidebar to run your "
            "first loan approval prediction. Every submission is automatically "
            "logged here. Running **🤝 Model Agreement** with \"Save to DB\" "
            "selected writes one row per model at the same timestamp."
        ),
        "Tip: Use the Prediction or Model Agreement pages to generate audit entries."
    )

    st.stop()


# ============================================================
# STATS TOP-LEVEL KPIs
# ============================================================

total = len(history)
approved_count = int((history["prediction"] == "Approved").sum())
rejected_count = int((history["prediction"] == "Rejected").sum())
approval_rate = approved_count / total * 100 if total else 0.0
avg_prob = history["probability"].mean() * 100 if total else 0.0
median_prob = history["probability"].median() * 100 if total else 0.0
prob_std = history["probability"].std() * 100 if total else 0.0

# Unique applicants: total timestamp groups
n_sessions = history["timestamp"].nunique() if total else 0
n_models_used = history["model"].nunique() if total else 0

# Date range
min_ts = history["timestamp_dt"].min() if total and "timestamp_dt" in history else None
max_ts = history["timestamp_dt"].max() if total and "timestamp_dt" in history else None

section_title("📊 Prediction Statistics (Overview)")

s_k1, s_k2, s_k3, s_k4 = st.columns(4)
s_k1.metric("Total Predictions", f"{total:,}", f"{n_sessions:,} unique sessions")
s_k2.metric(
    "Approved",
    f"{approved_count:,}",
    f"{approval_rate:.1f}% approval rate"
)
s_k3.metric(
    "Rejected",
    f"{rejected_count:,}",
    f"{rejected_count/total*100:.1f}% reject rate"
)
s_k4.metric(
    "Models used",
    f"{n_models_used}",
    f"{history['model'].value_counts().idxmax()} most used"
)

s2_k1, s2_k2, s2_k3 = st.columns(3)
s2_k1.metric(
    "Avg Approval Probability",
    f"{avg_prob:.1f} %",
    f"± {prob_std:.1f} σ"
)
s2_k2.metric(
    "Median Approval Probability",
    f"{median_prob:.1f} %"
)
s2_k3.metric(
    "Date range",
    (
        max_ts.strftime("%d %b %Y %H:%M")
        if pd.notna(max_ts) else "N/A"
    ),
    (
        "From " + min_ts.strftime("%d %b %Y")
        if pd.notna(min_ts) else None
    )
)

st.markdown("")

# ============================================================
# FILTERS + SEARCH
# ============================================================

section_title("🔎 Search / Filter History")

f_c1, f_c2, f_c3 = st.columns([1.1, 1, 1])

with f_c1:

    st.markdown("**Date/Time**")

    ts_df = history[history["timestamp_dt"].notna()].copy()
    if ts_df.empty:
        ts_min = datetime.now() - timedelta(days=1)
        ts_max = datetime.now()
    else:
        ts_min = ts_df["timestamp_dt"].min().to_pydatetime()
        ts_max = ts_df["timestamp_dt"].max().to_pydatetime()

    date_range = st.date_input(
        "Date range",
        value=(ts_min.date(), ts_max.date()),
        label_visibility="visible"
    )

    include_missing_ts = st.checkbox(
        "Include rows with missing timestamp",
        value=True
    )

with f_c2:

    st.markdown("**Prediction / Probability**")

    pred_filter = st.multiselect(
        "Prediction Outcome",
        options=["Approved", "Rejected"],
        default=["Approved", "Rejected"]
    )

    prob_min, prob_max = st.slider(
        "Approval Probability (%)",
        min_value=0, max_value=100,
        value=(0, 100)
    )

    dti_min, dti_max = st.slider(
        "Payment-to-Income DTI (%)",
        min_value=0, max_value=300, value=(0, 300)
    )

with f_c3:

    st.markdown("**Model & Profile**")

    model_options = sorted(
        history["model"].dropna().astype(str).unique().tolist()
    )
    model_filter = st.multiselect(
        "Model(s)",
        options=model_options,
        default=model_options
    )

    edu_options = sorted(
        history["education"].dropna().astype(str).unique().tolist()
    )
    edu_filter = st.multiselect(
        "Education",
        options=edu_options,
        default=edu_options
    )

    area_options = sorted(
        history["property_area"].dropna().astype(str).unique().tolist()
    )
    area_filter = st.multiselect(
        "Property Area",
        options=area_options,
        default=area_options
    )

search_c1, search_c2 = st.columns([3, 1])

with search_c1:
    search_term = st.text_input(
        "🔎 Free-text search (in Model, Prediction, Education, Area, or numeric range of Incomes / Loan Amount)",
        placeholder="e.g. XGBoost, Urban, Approved, Graduate…"
    ).strip()

with search_c2:
    credit_filter = st.selectbox(
        "Credit History",
        options=["Any", "1.0 – Good", "0.0 – Poor"],
        index=0
    )

# ---- Apply filters ----

filtered = history.copy()

# Date
if isinstance(date_range, tuple) and len(date_range) == 2:
    d_start, d_end = date_range
    start_dt = pd.Timestamp(d_start)
    end_dt = pd.Timestamp(d_end) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)

    if include_missing_ts:
        mask = (
            (filtered["timestamp_dt"].isna())
            | (
                (filtered["timestamp_dt"] >= start_dt)
                & (filtered["timestamp_dt"] <= end_dt)
            )
        )
    else:
        mask = (
            (filtered["timestamp_dt"] >= start_dt)
            & (filtered["timestamp_dt"] <= end_dt)
        )
    filtered = filtered[mask]

# Outcome & probability
filtered = filtered[
    filtered["prediction"].isin(pred_filter) &
    (filtered["probability"] * 100 >= prob_min) &
    (filtered["probability"] * 100 <= prob_max)
]

# Payment-to-income DTI
if "debt_to_income" in filtered.columns:
    dti_pct = pd.to_numeric(filtered["debt_to_income"], errors="coerce") * 100
    filtered = filtered[dti_pct.between(dti_min, dti_max) | dti_pct.isna()]

# Model / education / area
filtered = filtered[
    filtered["model"].astype(str).isin(model_filter) &
    filtered["education"].astype(str).isin(edu_filter) &
    filtered["property_area"].astype(str).isin(area_filter)
]

# Credit history
if credit_filter == "1.0 – Good":
    filtered = filtered[filtered["credit_history"] == 1.0]
elif credit_filter == "0.0 – Poor":
    filtered = filtered[filtered["credit_history"] == 0.0]

# Free-text search
if search_term:
    t = search_term.lower()
    text_mask = (
        filtered["model"].astype(str).str.lower().str.contains(t, na=False) |
        filtered["prediction"].astype(str).str.lower().str.contains(t, na=False) |
        filtered["education"].astype(str).str.lower().str.contains(t, na=False) |
        filtered["property_area"].astype(str).str.lower().str.contains(t, na=False)
    )
    try:
        num_val = float(t)
        num_mask = (
            (filtered["applicant_income"].round(0) == num_val) |
            (filtered["total_income"].round(0) == num_val) |
            (filtered["loan_amount"].round(2) == num_val) |
            (filtered["loan_to_income"].round(4) == num_val)
        )
        text_mask = text_mask | num_mask
    except ValueError:
        pass

    filtered = filtered[text_mask]

st.caption(
    f"Showing **{len(filtered):,} of {total:,}** records. "
    "All stats, charts and the CSV download below are based on the filtered view."
)

st.markdown("---")

# ============================================================
# DOWNLOAD CSV
# ============================================================

section_title("💾 Download History as CSV")

download_df = filtered.copy()
if "timestamp_dt" in download_df.columns:
    download_df = download_df.drop(columns=["timestamp_dt"])
download_df["probability"] = (
    (download_df["probability"] * 100).round(2).astype(str) + " %"
)
download_df["loan_to_income"] = download_df["loan_to_income"].round(4)

dl_c1, dl_c2 = st.columns([3, 1])

dl_c1.download_button(
    "📥 Download CURRENTLY FILTERED rows as CSV",
    data=download_df.to_csv(index=False).encode("utf-8"),
    file_name="filtered_loan_prediction_history.csv",
    mime="text/csv",
    width='stretch'
)

full_df = history.copy().drop(columns=["timestamp_dt"], errors="ignore")
full_df["probability"] = (
    (full_df["probability"] * 100).round(2).astype(str) + " %"
)
full_df["loan_to_income"] = full_df["loan_to_income"].round(4)

dl_c2.download_button(
    "📥 Download FULL history (CSV)",
    data=full_df.to_csv(index=False).encode("utf-8"),
    file_name="all_loan_prediction_history.csv",
    mime="text/csv",
    width='stretch'
)

st.markdown("---")

# ============================================================
# PREDICTION STATISTICS (charts)
# ============================================================

section_title("📈 Prediction Statistics (Filtered)")

st_c1, st_c2 = st.columns(2)

with st_c1:

    st.markdown("**Outcome distribution**")
    outcome_counts = filtered["prediction"].value_counts()
    st.bar_chart(outcome_counts, width='stretch', height=320)

    if len(filtered):
        f_approved = int((filtered["prediction"] == "Approved").sum())
        f_rate = f_approved / len(filtered) * 100
        st.caption(
            f"Filtered approval rate: **{f_rate:.1f}%** "
            f"({f_approved} of {len(filtered)})"
        )

with st_c2:

    st.markdown("**Predictions per Model**")
    m_counts = filtered["model"].value_counts().sort_values(ascending=True)
    st.bar_chart(m_counts, width='stretch', horizontal=True, height=340)


st_charts2_c1, st_charts2_c2 = st.columns(2)

with st_charts2_c1:

    st.markdown("**Approval Probability Distribution**")
    prob_bins = pd.cut(
        filtered["probability"] * 100,
        bins=[0, 20, 40, 50, 60, 80, 100],
        labels=["0-20", "20-40", "40-50", "50-60", "60-80", "80-100"],
        include_lowest=True
    )
    st.bar_chart(prob_bins.value_counts().sort_index(), width='stretch', height=300)

with st_charts2_c2:

    st.markdown("**Avg Probability by Model (descending)**")
    m_avg = (
        filtered.groupby("model")["probability"].mean() * 100
    ).round(1).sort_values(ascending=True)
    st.bar_chart(m_avg, horizontal=True, width='stretch', height=300, color="#0ea5e9")


st_extra_c1, st_extra_c2 = st.columns(2)

with st_extra_c1:

    st.markdown("**Approval Rate by Property Area**")
    area_pct = (
        filtered.groupby("property_area")["prediction"]
        .apply(lambda s: (s == "Approved").mean() * 100)
        .round(1)
        .sort_values(ascending=True)
    )
    st.bar_chart(area_pct, horizontal=True, width='stretch', height=260, color="#22c55e")

with st_extra_c2:

    st.markdown("**Approval Rate by Credit History**")
    ch_pct = (
        filtered.assign(
            Credit_History_Label=filtered["credit_history"].map(
                {1.0: "Good (1.0)", 0.0: "Poor (0.0)", np.nan: "Missing"}
            )
        )
        .groupby("Credit_History_Label")["prediction"]
        .apply(lambda s: (s == "Approved").mean() * 100)
        .round(1)
    )
    st.bar_chart(ch_pct, width='stretch', height=260, color="#8b5cf6")


st.markdown("")

with st.expander("📋 Aggregate numeric summary (filtered)", expanded=False):

    numeric_cols = [
        "applicant_income", "coapplicant_income",
        "total_income", "loan_amount",
        "debt_to_income", "probability"
    ]
    rename_map = {
        "applicant_income": "Applicant Income",
        "coapplicant_income": "Coapplicant Income",
        "total_income": "Total Income",
        "loan_amount": "Loan Amount",
        "debt_to_income": "Payment-to-Income DTI",
        "probability": "Probability (0–1)"
    }

    desc = (
        filtered[numeric_cols]
        .describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
        .T
        .rename(index=rename_map)
        .round(3)
    )

    st.dataframe(desc, width='stretch', height=340)

st.markdown("---")

# ============================================================
# FULL RECORDS TABLE
# ============================================================

section_title("🗂️ Prediction Records (Filtered)")

display_df = filtered.copy()

display_df["probability %"] = (display_df["probability"] * 100).round(2)
if "debt_to_income" in display_df.columns:
    display_df["Payment-to-Income DTI (%)"] = (display_df["debt_to_income"] * 100).round(2)

for col in ("applicant_income", "coapplicant_income", "total_income", "loan_amount"):
    if col in display_df.columns:
        display_df[col] = display_df[col].round(2)

display_columns = [
    "id", "timestamp", "model",
    "prediction", "probability %",
    "applicant_income", "coapplicant_income",
    "total_income", "loan_amount",
    "Payment-to-Income DTI (%)", "credit_history",
    "education", "property_area"
]
display_columns = [c for c in display_columns if c in display_df.columns]

st.dataframe(
    display_df[display_columns].style.apply(
        lambda row: [
            (
                "background:#dcfce7;font-weight:700;"
                if row.get("prediction") == "Approved"
                else "background:#fee2e2;font-weight:700;"
            )
            if c == "prediction" else ""
            for c in display_columns
        ],
        axis=1
    ),
    width='stretch',
    height=500,
    hide_index=True
)

st.markdown("---")

# ============================================================
# EXPANDABLE INDIVIDUAL RECORD CARDS
# ============================================================

section_title("📋 Individual Records (Full Detail)")

st.caption(
    "Expand any record below to see the full applicant snapshot with verdict badges, "
    "derived financials, and model tag."
)

def render_row_card(i, row):

    outcome = row.get("prediction", "")
    outcome_text = {
        "Approved": "✅ RECOMMEND APPROVAL",
        "Rejected": "❌ NOT RECOMMENDED",
    }.get(outcome, str(outcome).upper())

    ch_val = row.get("credit_history", np.nan)
    if pd.isna(ch_val):
        credit_label = "Missing"
    elif ch_val == 1.0:
        credit_label = "Good (1.0)"
    else:
        credit_label = "Poor (0.0)"

    prob_pct = float(row.get("probability", 0.0) or 0) * 100

    dti = pd.to_numeric(pd.Series([row.get("debt_to_income")]), errors="coerce").iloc[0]
    dti_label = f"{dti * 100:.2f}%" if pd.notna(dti) else "Unavailable (older record)"

    title = (
        f"#{int(row.get('id', i))} · {row.get('timestamp','N/A')}"
        f" · {row.get('model','N/A')}"
    )

    with st.expander(title, expanded=(i == 1)):
        st.subheader(outcome_text)
        st.metric(
            "Approval Probability",
            f"{prob_pct:.2f}%",
            delta=("Approval band" if prob_pct >= 60 else ("Review band" if prob_pct >= 40 else "Not-recommended band"))
        )
        st.caption(f"Credit History: {credit_label}")
        basis = row.get("decision_basis")
        if pd.isna(basis) or not str(basis).strip():
            basis = "A detailed basis was not saved for this older prediction."
        st.info(f"**Approval status: {outcome}**\n\n**Decision basis:** {basis}")

        info_col, decision_col = st.columns(2)
        with info_col:
            st.markdown("**👤 Applicant Information**")
            applicant_info = pd.DataFrame(
                [
                    ("Education", row.get("education", "N/A")),
                    ("Property Area", row.get("property_area", "N/A")),
                    ("Applicant Income", f"{row.get('applicant_income', 0):,.2f}"),
                    ("Co-applicant Income", f"{row.get('coapplicant_income', 0):,.2f}"),
                    ("Total Income", f"{row.get('total_income', 0):,.2f}"),
                    ("Marital Status (context)", row.get("marital_status", "N/A")),
                ],
                columns=["Detail", "Value"]
            ).astype("string")
            st.dataframe(applicant_info, width="stretch", hide_index=True)

        with decision_col:
            st.markdown("**🏦 Loan & Prediction**")
            loan_info = pd.DataFrame(
                [
                    ("Model", row.get("model", "N/A")),
                    ("Verdict", outcome),
                    ("Probability", f"{prob_pct:.2f}%"),
                    ("Loan Amount", f"{row.get('loan_amount', 0):,.2f}"),
                    ("Payment-to-Income DTI", dti_label),
                    ("Estimated Monthly Payment", f"{row.get('estimated_monthly_payment', 0):,.2f}"),
                    ("Existing Monthly Debt", f"{row.get('existing_monthly_debt', 0):,.2f}"),
                    ("Affordability Rate", f"{row.get('annual_interest_rate', 0):,.2f}%"),
                ],
                columns=["Detail", "Value"]
            ).astype("string")
            st.dataframe(loan_info, width="stretch", hide_index=True)

# Render at most 50 expanded entries to keep UI snappy
render_limit = min(50, len(filtered))
for i, (_, row) in enumerate(filtered.head(render_limit).iterrows(), start=1):
    render_row_card(i, row)

if len(filtered) > render_limit:
    render_alert(
        f"Only first {render_limit} of {len(filtered):,} filtered records have "
        "expandable cards to keep the page responsive. Use the full table above "
        "or apply filters (search / date-range / probability-slider / model) to "
        "narrow in, then the expandables will refresh for those rows.",
        kind="info"
    )

st.markdown("")

render_alert(
    """
    <b>💡 Audit & reproducibility tips:</b><br><br>
    1. Click <b>Download CURRENTLY FILTERED rows as CSV</b> to hand exactly the
       visible records to an auditor.<br>
    2. Filter by "Model = XGBoost Tuned" + date range = last month to review
       only the production-candidate model predictions.<br>
    3. Scroll the Probability slider to "50–100" Approved, then filter by
       "Credit History = Good" to confirm intuition matches the stats at the
       top of the page.<br>
    4. Use <b>🤝 Model Agreement</b> with "Save to DB" checked to produce one
       row per model at the same timestamp for direct comparison in this page.
    """,
    kind="info"
)


# ============================================================
# DISCLAIMER FOOTER
# ============================================================

