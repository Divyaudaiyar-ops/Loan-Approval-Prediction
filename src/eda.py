import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Load dataset
df = pd.read_csv("data/loan_data.csv")

# Create folder for graphs
output_dir = Path("data/eda_graphs")
output_dir.mkdir(parents=True, exist_ok=True)

print("=" * 50)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 50)

print("\nDataset Shape:", df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:", df.duplicated().sum())

print("\nLoan Status Distribution:")
print(df["Loan_Status"].value_counts())

# -----------------------------
# 1. LOAN STATUS
# -----------------------------

plt.figure(figsize=(7, 5))

sns.countplot(data=df, x="Loan_Status")

plt.title("Loan Approval Status Distribution")
plt.xlabel("Loan Status")
plt.ylabel("Number of Applicants")

plt.tight_layout()
plt.savefig(output_dir / "01_loan_status.png", dpi=300)
plt.close()


# -----------------------------
# 2. APPLICANT INCOME
# -----------------------------

plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="ApplicantIncome",
    kde=True
)

plt.title("Applicant Income Distribution")
plt.xlabel("Applicant Income")
plt.ylabel("Frequency")

plt.tight_layout()
plt.savefig(output_dir / "02_applicant_income.png", dpi=300)
plt.close()


# -----------------------------
# 3. LOAN AMOUNT
# -----------------------------

plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="LoanAmount",
    kde=True
)

plt.title("Loan Amount Distribution")
plt.xlabel("Loan Amount")
plt.ylabel("Frequency")

plt.tight_layout()
plt.savefig(output_dir / "03_loan_amount.png", dpi=300)
plt.close()


# -----------------------------
# 4. CREDIT HISTORY
# -----------------------------

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="Credit_History",
    hue="Loan_Status"
)

plt.title("Credit History vs Loan Approval")
plt.xlabel("Credit History")
plt.ylabel("Number of Applicants")

plt.tight_layout()
plt.savefig(output_dir / "04_credit_history.png", dpi=300)
plt.close()


# -----------------------------
# 5. EDUCATION
# -----------------------------

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="Education",
    hue="Loan_Status"
)

plt.title("Education vs Loan Approval")
plt.xlabel("Education")
plt.ylabel("Number of Applicants")

plt.tight_layout()
plt.savefig(output_dir / "05_education.png", dpi=300)
plt.close()


# -----------------------------
# 6. PROPERTY AREA
# -----------------------------

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="Property_Area",
    hue="Loan_Status"
)

plt.title("Property Area vs Loan Approval")
plt.xlabel("Property Area")
plt.ylabel("Number of Applicants")

plt.tight_layout()
plt.savefig(output_dir / "06_property_area.png", dpi=300)
plt.close()


# -----------------------------
# 7. CORRELATION HEATMAP
# -----------------------------

numeric_df = df.select_dtypes(include="number")

plt.figure(figsize=(10, 7))

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Feature Correlation Heatmap")

plt.tight_layout()
plt.savefig(output_dir / "07_correlation_heatmap.png", dpi=300)
plt.close()


print("\nEDA completed successfully!")
print("Graphs saved in:")
print(output_dir)