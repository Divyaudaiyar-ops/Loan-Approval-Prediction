import streamlit as st
import pandas as pd
import numpy as np
import os
import sys


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
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Data Analytics · Loan Approval DSS",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

inject_theme_css()
render_sidebar_brand(
    app_name="Data Analytics",
    subtitle="Dataset EDA & Feature Insights"
)


# ============================================================
# FILE PATHS
# ============================================================

DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "loan_data.csv"
)


# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_dataset():

    if not os.path.exists(DATA_PATH):

        render_error_state(
            f"Dataset not found:\n{DATA_PATH}",
            hint="Ensure data/loan_data.csv exists in the project root."
        )

        st.stop()

    return pd.read_csv(DATA_PATH)


with loading_spinner("Loading historical loan dataset …"):
    df = load_dataset()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    section_title("📈 Data Analytics")
    st.markdown(
        """
        Explore distributions,
        correlations, missing
        values, cross-tabs and
        EDA visuals of the
        historical loan dataset.
        """
    )

    st.divider()

    st.markdown("### ⚙️ Options")

    drop_outliers = st.checkbox(
        "Drop extreme outliers (> 99th percentile) for scatter plots",
        value=False,
        help="Removes rows where ApplicantIncome or LoanAmount exceed their 99th percentile"
    )

    log_scale = st.checkbox(
        "Apply log scale to numeric distributions",
        value=False,
        help="Useful for right-skewed income / loan-amount histograms"
    )

    st.divider()

    st.caption(
        "Explainable ML Based Loan Approval Prediction"
    )

    st.caption(
        "Academic Decision Support Prototype"
    )


# ============================================================
# PAGE HERO
# ============================================================

render_page_hero(
    chip_label="EXPLORATORY ANALYSIS",
    title="Data Analytics",
    subtitle=(
        f"Exploratory analysis of the historical loan dataset "
        f"({len(df):,} applications · {df.shape[1]} columns). "
        "Use the sidebar controls to adjust outlier-handling and "
        "numeric-distribution scaling."
    ),
    mini_stats=[
        {"label": "Applications", "value": f"{len(df):,}", "sub": f"{df.shape[1]} features", "variant": "primary"},
        {"label": "Approval Rate", "value": f"{(df['Loan_Status'] == 'Y').mean() * 100:.1f}%" if "Loan_Status" in df.columns else "N/A", "sub": "Historical log", "variant": "success"},
        {"label": "Missing Cells", "value": f"{int(df.isna().sum().sum()):,}", "sub": "Across all columns", "variant": "warning"},
    ],
)


# ============================================================
# PREPROCESS HELPERS
# ============================================================

def encode_for_corr(d: pd.DataFrame) -> pd.DataFrame:
    out = d.copy()

    status_map = {"Y": 1, "N": 0}
    if "Loan_Status" in out.columns:
        out["Loan_Status"] = out["Loan_Status"].map(status_map)

    yesno_cols = [c for c in ["Married", "Education", "Self_Employed"]
                  if c in out.columns]
    for c in yesno_cols:
        if c == "Education":
            out[c] = out[c].map({"Graduate": 1, "Not Graduate": 0})
        else:
            out[c] = out[c].map({"Yes": 1, "No": 0})

    if "Gender" in out.columns:
        out["Gender"] = out["Gender"].map({"Male": 1, "Female": 0})

    if "Property_Area" in out.columns:
        area_ord = {"Rural": 0, "Semiurban": 1, "Urban": 2}
        out["Property_Area"] = out["Property_Area"].map(area_ord)

    if "Dependents" in out.columns:
        out["Dependents"] = (
            out["Dependents"].astype(str).str.replace("3+", "3")
            .astype(float)
        )

    num_cols = [
        c for c in out.columns if c != "Loan_ID"
        and pd.api.types.is_numeric_dtype(out[c])
    ]

    return out[num_cols]


def safe_log(series: pd.Series) -> pd.Series:
    return np.log1p(series.fillna(0))


# ============================================================
# 1. DATASET STATISTICS (TOP-LEVEL)
# ============================================================

section_title("📊 Dataset Statistics")

ds_k1, ds_k2, ds_k3, ds_k4, ds_k5 = st.columns(5)

total_rows = len(df)
total_cols = df.shape[1]

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
categorical_cols = df.select_dtypes(include=object).columns.tolist()

missing_cells = int(df.isna().sum().sum())
missing_share = missing_cells / (total_rows * total_cols) * 100

approved_share = (df["Loan_Status"] == "Y").mean() * 100 if "Loan_Status" in df.columns else 0.0

ds_k1.markdown(
    f"""
    <div class="stat-card">
        <div class="stat-head">Applications (Rows)</div>
        <div class="stat-val">{total_rows:,}</div>
        <div class="stat-sub">Historical loan applications</div>
    </div>
    """,
    unsafe_allow_html=True
)

ds_k2.markdown(
    f"""
    <div class="stat-card">
        <div class="stat-head">Features (Columns)</div>
        <div class="stat-val">{total_cols}</div>
        <div class="stat-sub">
            {len(numeric_cols)} numeric · {len(categorical_cols)} categorical
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

ds_k3.markdown(
    f"""
    <div class="stat-card">
        <div class="stat-head">Missing Cells</div>
        <div class="stat-val">{missing_cells:,}</div>
        <div class="stat-sub">{missing_share:.2f} % of all cells</div>
    </div>
    """,
    unsafe_allow_html=True
)

ds_k4.markdown(
    f"""
    <div class="stat-card">
        <div class="stat-head">Approval Rate</div>
        <div class="stat-val">{approved_share:.1f} %</div>
        <div class="stat-sub">
            {int((df['Loan_Status'] == 'Y').sum())} Approved ·
            {int((df['Loan_Status'] == 'N').sum())} Rejected
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

avg_loan = df["LoanAmount"].mean() if "LoanAmount" in df.columns else np.nan
avg_income = df["ApplicantIncome"].mean() if "ApplicantIncome" in df.columns else np.nan
ds_k5.markdown(
    f"""
    <div class="stat-card">
        <div class="stat-head">Averages</div>
        <div class="stat-val" style="font-size:16px;">
            {avg_income:,.0f} Income · {avg_loan:,.0f} Loan
        </div>
        <div class="stat-sub">
            ApplicantIncome & LoanAmount
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("")

dsc_c1, dsc_c2 = st.columns([1.4, 1])

with dsc_c1:

    st.subheader("Feature Types")

    types_df = pd.DataFrame({
        "Feature": df.columns,
        "Data Type": [str(t) for t in df.dtypes.values],
        "Unique Values": [
            df[c].nunique(dropna=True) for c in df.columns
        ],
        "Missing Count": [
            int(df[c].isna().sum()) for c in df.columns
        ],
        "Missing %": [
            round(df[c].isna().mean() * 100, 2) for c in df.columns
        ]
    })

    st.dataframe(
        types_df.style.bar(
            subset=["Missing %"], color="#fecaca", vmin=0, vmax=25
        ),
        width='stretch',
        hide_index=True,
        height=30 + 35 * len(types_df)
    )

with dsc_c2:

    st.subheader("Numeric Column Summary")

    num_desc = df[numeric_cols].describe().T.round(2).reset_index()
    num_desc = num_desc.rename(columns={"index": "Feature"})
    st.dataframe(
        num_desc,
        width='stretch',
        hide_index=True,
        height=310
    )

st.markdown("---")

# ============================================================
# 2. LOAN STATUS DISTRIBUTION
# ============================================================

section_title("🏁 Loan Status Distribution")

status_counts_raw = df["Loan_Status"].value_counts()
status_counts = status_counts_raw.rename({"Y": "Approved", "N": "Rejected"})
status_pct = (status_counts / status_counts.sum() * 100).round(1)

st_k1, st_k2 = st.columns(2)

st_k1.metric(
    "Approved Applications",
    f"{int(status_counts['Approved']):,}",
    f"{status_pct['Approved']:.1f} % of all applications"
)
st_k2.metric(
    "Rejected Applications",
    f"{int(status_counts['Rejected']):,}",
    f"{status_pct['Rejected']:.1f} % of all applications"
)

st.markdown("")

status_c1, status_c2 = st.columns([1.2, 1])

with status_c1:

    st.markdown("**Loan Status Count**")
    st.bar_chart(
        pd.DataFrame(status_counts).rename(columns={
            "Loan_Status": "Applications"
        }),
        width='stretch',
        height=340,
        color="#1f77b4"
    )

with status_c2:

    st.markdown("**Approval / Rejection Share**")
    st.bar_chart(
        pd.DataFrame(status_pct).rename(columns={
            "Loan_Status": "% of Applications"
        }),
        width='stretch',
        horizontal=True,
        height=340,
        color="#22c55e" if status_pct.get("Approved", 0) > 50 else "#ef4444"
    )

with st.expander(
    "📋 Cross-tabs by Loan Status",
    expanded=False
):

    st.markdown(
        "Breakdown count of each categorical feature split by Approved / Rejected."
    )

    cat_feat = st.multiselect(
        "Features to cross-tab",
        [c for c in categorical_cols if c not in ("Loan_ID", "Loan_Status")],
        default=["Gender", "Married", "Dependents"]
    )

    for feat in cat_feat:

        st.markdown(f"**{feat} × Loan Status**")

        ct = pd.crosstab(df[feat].fillna("(Missing)"), df["Loan_Status"])
        ct = ct.rename(columns={"Y": "Approved", "N": "Rejected"})
        ct["% Approved"] = (
            ct["Approved"] / (ct["Approved"] + ct["Rejected"]) * 100
        ).round(1)

        st.dataframe(ct, width='stretch', height=160)

st.markdown("---")

# ============================================================
# 3. INCOME DISTRIBUTION
# ============================================================

section_title("💵 Income Distribution")

income_features = [
    c for c in ("ApplicantIncome", "CoapplicantIncome")
    if c in df.columns
]

inc_c1, inc_c2 = st.columns([1.1, 1])

with inc_c1:

    st.markdown("**Applicant Income by Status**")

    inc_df = df[["ApplicantIncome", "Loan_Status"]].dropna().copy()

    if drop_outliers:
        cap = inc_df["ApplicantIncome"].quantile(0.99)
        inc_df = inc_df[inc_df["ApplicantIncome"] <= cap]

    if log_scale:
        inc_df["ApplicantIncome"] = safe_log(inc_df["ApplicantIncome"])

    inc_df["Status"] = inc_df["Loan_Status"].map(
        {"Y": "Approved", "N": "Rejected"}
    )

    bins = [0, 2500, 5000, 10000, 20000, df["ApplicantIncome"].max() + 1]
    labels = ["0-2.5k", "2.5k-5k", "5k-10k", "10k-20k", "20k+"]

    if log_scale:
        bins = np.log1p(bins)

    inc_df["Income Band"] = pd.cut(
        inc_df["ApplicantIncome"],
        bins=bins,
        labels=labels,
        include_lowest=True
    )

    band_pivot = pd.crosstab(inc_df["Income Band"], inc_df["Status"])
    st.bar_chart(band_pivot, width='stretch', height=360)

with inc_c2:

    st.markdown("**Income Statistics (by Status)**")

    inc_stats = (
        df.groupby("Loan_Status")[income_features]
        .describe()
        .T
        .round(2)
        .rename(columns={"Y": "Approved", "N": "Rejected"})
    )
    st.dataframe(inc_stats, width='stretch', height=360)

st.markdown("")

inc_sub_c1, inc_sub_c2 = st.columns(2)

with inc_sub_c1:
    st.markdown("**Co-applicant Income Histogram**")
    co_series = df["CoapplicantIncome"].dropna()
    if drop_outliers:
        co_series = co_series[co_series <= co_series.quantile(0.99)]
    if log_scale:
        co_series = safe_log(co_series)
    co_bins = pd.cut(co_series, bins=10)
    st.bar_chart(co_bins.value_counts().sort_index(), width='stretch', height=300)

with inc_sub_c2:
    st.markdown("**Total Household Income (Applicant + Co-applicant)**")
    total_income = df["ApplicantIncome"].fillna(0) + df["CoapplicantIncome"].fillna(0)
    if drop_outliers:
        cap = total_income.quantile(0.99)
        total_income = total_income[total_income <= cap]
    if log_scale:
        total_income = safe_log(total_income)
    total_bins = pd.cut(total_income, bins=12)
    st.bar_chart(total_bins.value_counts().sort_index(), width='stretch', height=300)

st.markdown("---")

# ============================================================
# 4. LOAN AMOUNT DISTRIBUTION
# ============================================================

section_title("💰 Loan Amount Distribution")

ln_c1, ln_c2 = st.columns([1.2, 1])

with ln_c1:

    st.markdown("**Loan Amount (by Loan Status)**")

    loan_df = df[["LoanAmount", "Loan_Status"]].dropna().copy()

    if drop_outliers:
        cap = loan_df["LoanAmount"].quantile(0.99)
        loan_df = loan_df[loan_df["LoanAmount"] <= cap]

    if log_scale:
        loan_df["LoanAmount"] = safe_log(loan_df["LoanAmount"])

    loan_bins = [0, 50, 100, 200, 400, df["LoanAmount"].max() + 1]
    loan_labels = ["0-50", "50-100", "100-200", "200-400", "400+"]

    if log_scale:
        loan_bins = np.log1p(loan_bins)

    loan_df["Loan Band"] = pd.cut(
        loan_df["LoanAmount"],
        bins=loan_bins,
        labels=loan_labels,
        include_lowest=True
    )
    loan_df["Status"] = loan_df["Loan_Status"].map(
        {"Y": "Approved", "N": "Rejected"}
    )

    loan_pivot = pd.crosstab(loan_df["Loan Band"], loan_df["Status"])
    st.bar_chart(loan_pivot, width='stretch', height=360)

with ln_c2:

    st.markdown("**Loan Amount Term Distribution**")

    term_vc = df["Loan_Amount_Term"].dropna().astype(int).value_counts().sort_index()

    term_df = pd.DataFrame({
        "Months": term_vc.index.astype(str),
        "Applications": term_vc.values
    }).set_index("Months")

    st.bar_chart(term_df, width='stretch', height=330, color="#f59e0b")

    with st.expander("Loan Amount — Full Descriptive Statistics", expanded=False):
        st.dataframe(
            df[["LoanAmount", "Loan_Amount_Term"]].describe().round(2),
            width='stretch'
        )

st.markdown("---")

# ============================================================
# 5. CREDIT HISTORY ANALYSIS
# ============================================================

section_title("💳 Credit History Analysis")

ch_df = df.copy()
ch_df["Credit_History_Label"] = ch_df["Credit_History"].map(
    {1.0: "1.0 – Good / Meets guidelines", 0.0: "0.0 – Poor / Absent", np.nan: "Missing"}
)
ch_df["Loan_Status_Label"] = ch_df["Loan_Status"].map(
    {"Y": "Approved", "N": "Rejected"}
)

ch_ct = pd.crosstab(
    ch_df["Credit_History_Label"],
    ch_df["Loan_Status_Label"]
)

ch_ct["% Approved"] = (
    (ch_ct["Approved"] / (ch_ct["Approved"] + ch_ct["Rejected"]) * 100)
    .fillna(0)
    .round(1)
)

ch_c1, ch_c2 = st.columns([1.1, 1])

with ch_c1:
    st.markdown("**Approval vs Credit History**")
    st.bar_chart(
        ch_ct[["Approved", "Rejected"]],
        width='stretch',
        height=340
    )

with ch_c2:
    st.markdown("**Approval Rate per Credit-History Bucket**")
    st.dataframe(ch_ct, width='stretch', height=240)
    st.caption(
        "Credit History is typically the single strongest predictor of "
        "Loan_Status in this dataset — a 1.0 lifts approval drastically."
    )

st.markdown("")

ch_sub = st.columns(3)

# Missing-value credit-history approval rate
missing_ch = ch_df[ch_df["Credit_History"].isna()]
poor_ch = ch_df[ch_df["Credit_History"] == 0.0]
good_ch = ch_df[ch_df["Credit_History"] == 1.0]

def approval_pct(sub):
    if sub.empty:
        return 0.0
    return (sub["Loan_Status"] == "Y").mean() * 100

ch_sub[0].metric(
    "No Credit History (missing)",
    f"{len(missing_ch):,}",
    f"{approval_pct(missing_ch):.1f}% Approved"
)
ch_sub[1].metric(
    "Poor Credit History (0.0)",
    f"{len(poor_ch):,}",
    f"{approval_pct(poor_ch):.1f}% Approved"
)
ch_sub[2].metric(
    "Good Credit History (1.0)",
    f"{len(good_ch):,}",
    f"{approval_pct(good_ch):.1f}% Approved"
)

st.markdown("---")

# ============================================================
# 6. EDUCATION ANALYSIS
# ============================================================

section_title("🎓 Education Analysis")

edu_ct = pd.crosstab(df["Education"].fillna("(Missing)"), df["Loan_Status"])
edu_ct = edu_ct.rename(columns={"Y": "Approved", "N": "Rejected"})
edu_ct["% Approved"] = (
    edu_ct["Approved"] / (edu_ct["Approved"] + edu_ct["Rejected"]) * 100
).round(1)

edu_ct["Total Applications"] = edu_ct["Approved"] + edu_ct["Rejected"]

edu_c1, edu_c2 = st.columns([1.2, 1])

with edu_c1:
    st.markdown("**Education Level × Loan Status**")
    st.bar_chart(edu_ct[["Approved", "Rejected"]], width='stretch', height=320)

with edu_c2:
    st.markdown("**Approval Rate by Education**")
    st.dataframe(edu_ct, width='stretch', height=240)

st.markdown("")

edu_income_c1, edu_income_c2 = st.columns(2)

with edu_income_c1:
    st.markdown("**Median Applicant Income by Education**")
    inc_edu = (
        df.groupby("Education")["ApplicantIncome"]
        .median()
        .round(0)
        .sort_values(ascending=True)
    )
    st.bar_chart(inc_edu, horizontal=True, width='stretch', height=260)

with edu_income_c2:
    st.markdown("**Median Loan Amount by Education**")
    loan_edu = (
        df.groupby("Education")["LoanAmount"]
        .median()
        .round(1)
        .sort_values(ascending=True)
    )
    st.bar_chart(loan_edu, horizontal=True, width='stretch', height=260, color="#0ea5e9")

st.markdown("---")

# ============================================================
# 7. PROPERTY AREA ANALYSIS
# ============================================================

section_title("📍 Property Area Analysis")

area_ct = pd.crosstab(df["Property_Area"].fillna("(Missing)"), df["Loan_Status"])
area_ct = area_ct.rename(columns={"Y": "Approved", "N": "Rejected"})
area_ct["% Approved"] = (
    area_ct["Approved"] / (area_ct["Approved"] + area_ct["Rejected"]) * 100
).round(1)

area_ct["Total"] = area_ct["Approved"] + area_ct["Rejected"]

area_c1, area_c2 = st.columns([1.2, 1])

with area_c1:
    st.markdown("**Property Area × Loan Status**")
    st.bar_chart(area_ct[["Approved", "Rejected"]], width='stretch', height=320)

with area_c2:
    st.markdown("**Approval Rate by Property Area**")
    st.dataframe(area_ct, width='stretch', height=240)

st.markdown("")

area_extra_c1, area_extra_c2, area_extra_c3 = st.columns(3)

area_extra_c1.metric(
    "Best-performing Area",
    area_ct["% Approved"].idxmax(),
    f"{area_ct['% Approved'].max():.1f}% Approved"
)
area_extra_c2.metric(
    "Most Applications",
    area_ct["Total"].idxmax(),
    f"{int(area_ct['Total'].max()):,} applications"
)

area_ltv_df = df.assign(
    Loan_to_Income=lambda d: d["LoanAmount"] / (d["ApplicantIncome"].fillna(0) + d["CoapplicantIncome"].fillna(0) + 1)
)
avg_ltv_by_area = (
    area_ltv_df.groupby("Property_Area")["Loan_to_Income"].median().round(4)
)

area_extra_c3.metric(
    "Highest Median LTI",
    avg_ltv_by_area.idxmax(),
    f"{avg_ltv_by_area.max():.4f}"
)

st.markdown("---")

# ============================================================
# 8. INCOME vs LOAN AMOUNT
# ============================================================

section_title("📐 Income vs Loan Amount")

scatter_df = df[[
    "ApplicantIncome", "CoapplicantIncome",
    "LoanAmount", "Loan_Status"
]].dropna().copy()

scatter_df["Total Household Income"] = (
    scatter_df["ApplicantIncome"] + scatter_df["CoapplicantIncome"]
)

if drop_outliers:
    for col in ["ApplicantIncome", "Total Household Income", "LoanAmount"]:
        cap = scatter_df[col].quantile(0.99)
        scatter_df = scatter_df[scatter_df[col] <= cap]

scatter_c1, scatter_c2 = st.columns(2)

with scatter_c1:
    st.markdown("**Applicant Income × Loan Amount (color = Status)**")

    scatter_df["Status Color"] = scatter_df["Loan_Status"].map(
        {"Y": 1.0, "N": 0.0}
    )

    st.scatter_chart(
        scatter_df,
        x="ApplicantIncome",
        y="LoanAmount",
        color="Status Color",
        size=40,
        width='stretch',
        height=380
    )
    st.caption(
        "🟢 = Approved · 🔴 = Rejected. "
        "Positive correlation is expected: higher income → larger loans requested/approved."
    )

with scatter_c2:

    st.markdown("**Total Household Income × Loan Amount**")

    st.scatter_chart(
        scatter_df,
        x="Total Household Income",
        y="LoanAmount",
        color="Status Color",
        size=40,
        width='stretch',
        height=380
    )

st.markdown("")

lt_c1, lt_c2 = st.columns(2)

with lt_c1:
    st.markdown("**Loan-to-Income Ratio Distribution by Status**")

    scatter_df["LTI"] = scatter_df["LoanAmount"] / (
        scatter_df["Total Household Income"] + 1
    )
    scatter_df["LTI Band"] = pd.cut(
        scatter_df["LTI"],
        bins=[0, 0.02, 0.04, 0.08, scatter_df["LTI"].max() + 0.01],
        labels=["<0.02 (Low)", "0.02-0.04", "0.04-0.08", ">0.08 (High)"],
        include_lowest=True
    )

    lti_ct = pd.crosstab(
        scatter_df["LTI Band"],
        scatter_df["Loan_Status"].map({"Y": "Approved", "N": "Rejected"})
    )

    st.bar_chart(lti_ct, width='stretch', height=300)

with lt_c2:

    st.markdown("**% Approved by LTI Band**")

    pct_approved = (
        lti_ct["Approved"] / (lti_ct["Approved"] + lti_ct["Rejected"]) * 100
    ).fillna(0).round(1)

    st.bar_chart(pct_approved, width='stretch', height=300, color="#8b5cf6")
    st.caption(
        "Increasing LTI generally correlates with lower approval rates."
    )

st.markdown("---")

# ============================================================
# 9. CORRELATION HEATMAP
# ============================================================

section_title("🔥 Correlation Heatmap")

corr_df_raw = encode_for_corr(df).dropna()
corr_matrix = corr_df_raw.corr().round(3)

corr_c1, corr_c2 = st.columns([1.3, 1])

with corr_c1:

    st.markdown("**Pearson Correlation Matrix**")

    # Styled table heatmap
    def _heatmap(val):
        if pd.isna(val):
            return "background:#fff;"
        # Colormap: -1 blue → 0 white → +1 red
        sign = -1 if val < 0 else 1
        intensity = min(abs(val) * 180, 180)
        if sign > 0:
            r, g, b = 255, 255 - int(intensity * 0.6), 255 - intensity
        else:
            r, g, b = 255 - intensity, 255 - int(intensity * 0.6), 255
        return (
            f"background-color:rgb({int(r)},{int(g)},{int(b)});"
            f"font-weight:700;color:#111827;"
        )

    st.dataframe(
        corr_matrix.style.format("{:.2f}").applymap(_heatmap),
        width='stretch',
        height=34 + 36 * len(corr_matrix)
    )

with corr_c2:

    st.markdown("**Top 10 Absolute Correlations with Loan_Status**")

    if "Loan_Status" in corr_matrix.columns:
        target_corr = (
            corr_matrix["Loan_Status"]
            .drop(labels=["Loan_Status"], errors="ignore")
            .sort_values(key=lambda s: s.abs(), ascending=False)
            .head(10)
            .to_frame()
            .rename(columns={"Loan_Status": "Corr with Loan_Status"})
        )

        st.dataframe(
            target_corr.style.format("{:.3f}").applymap(_heatmap),
            width='stretch',
            height=380
        )
    else:
        st.info("Loan_Status target not in correlation matrix.")

st.markdown("")

with st.expander("📖 How to read the correlation heatmap", expanded=False):
    st.markdown(
        """
        - **+1** (deep red): perfect positive correlation.
        - **0** (white): no linear correlation.
        - **−1** (deep blue): perfect negative correlation.
        - The column **Top 10 Correlations with Loan_Status** is the quickest
          short-list of which features the ML models will lean on most heavily.
        """
    )

st.markdown("---")

# ============================================================
# 10. MISSING VALUE STATISTICS
# ============================================================

section_title("🧩 Missing-Value Statistics")

missing_count = df.isna().sum()
missing_pct = (df.isna().mean() * 100).round(2)

missing_summary = pd.DataFrame({
    "Feature": df.columns,
    "Missing Count": missing_count.values,
    "Missing %": missing_pct.values,
    "Total Values": len(df),
    "Non-Missing": len(df) - missing_count.values
})

missing_summary = missing_summary.sort_values(
    "Missing %", ascending=False
).reset_index(drop=True)

mv_c1, mv_c2, mv_c3 = st.columns([1.4, 1, 1])

with mv_c1:

    st.markdown("**Missingness by Feature**")

    st.dataframe(
        missing_summary.style.bar(
            subset=["Missing %"], color="#fecaca", vmin=0, vmax=max(missing_pct.max(), 5)
        ),
        width='stretch',
        hide_index=True,
        height=340
    )

with mv_c2:

    st.markdown("**Percentage Missing (Descending)**")

    missing_plot = (
        missing_summary[missing_summary["Missing %"] > 0]
        .set_index("Feature")[["Missing %"]]
        .sort_values("Missing %", ascending=True)
    )

    if not missing_plot.empty:
        st.bar_chart(
            missing_plot,
            horizontal=True,
            width='stretch',
            height=320,
            color="#ef4444"
        )
    else:
        st.success("No missing values in any column!")

with mv_c3:

    st.markdown("**Approval Rate when Missing**")

    findings = []
    for feat in df.columns[df.isna().any()]:
        mask = df[feat].isna()
        pct_when_missing = (
            (df.loc[mask, "Loan_Status"] == "Y").mean() * 100
            if mask.any() and "Loan_Status" in df.columns else None
        )
        pct_when_present = (
            (df.loc[~mask, "Loan_Status"] == "Y").mean() * 100
            if (~mask).any() and "Loan_Status" in df.columns else None
        )
        if pct_when_missing is not None:
            findings.append({
                "Feature": feat,
                "Approval if Missing %": round(pct_when_missing, 1),
                "Approval if Present %": round(pct_when_present, 1),
                "Δ (pp)": round(pct_when_missing - pct_when_present, 1)
            })

    if findings:
        findf = pd.DataFrame(findings).set_index("Feature")
        st.dataframe(findf, width='stretch', height=320)
    else:
        st.info("No missing values detected — approval-rate comparison skipped.")

st.markdown("")

mv_overall_c1, mv_overall_c2, mv_overall_c3, mv_overall_c4 = st.columns(4)

cols_with_missing = int((missing_count > 0).sum())
rows_with_any_missing = int(df.isna().any(axis=1).sum())
rows_complete = total_rows - rows_with_any_missing
max_miss_feat = missing_count.idxmax()
max_miss_val = int(missing_count.max())

mv_overall_c1.metric(
    "Columns with any missing value",
    f"{cols_with_missing} / {total_cols}"
)
mv_overall_c2.metric(
    "Rows with ≥ 1 missing cell",
    f"{rows_with_any_missing:,}",
    f"{rows_with_any_missing/total_rows*100:.1f} % of rows"
)
mv_overall_c3.metric(
    "Complete rows (no missing data)",
    f"{rows_complete:,}",
    f"{rows_complete/total_rows*100:.1f} % of rows"
)
mv_overall_c4.metric(
    "Largest per-column missing",
    f"{max_miss_val:,} cells",
    f"Column: {max_miss_feat}"
)

st.markdown("---")

st.info(
    """
    💡 **Tips from the EDA:**

    1. Credit History 1.0 typically dominates approval rates — if you see a
       huge difference between Good/Poor credit, keep an eye on that feature.
    2. ApplicantIncome and LoanAmount are right-skewed: log-transformation is
       often helpful, toggleable in the sidebar.
    3. Look at the **Top 10 correlations with Loan_Status** in the heatmap
       to confirm which numeric features the ML models will value most.
    """
)


render_disclaimer_footer()
