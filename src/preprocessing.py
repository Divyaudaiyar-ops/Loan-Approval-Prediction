import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

def load_data(path="data/loan_data.csv"):
    df = pd.read_csv(path)

    # Remove Loan_ID because it is only an identifier
    if "Loan_ID" in df.columns:
        df = df.drop("Loan_ID", axis=1)

    return df


# --------------------------------------------------
# FEATURE ENGINEERING
# --------------------------------------------------

def add_features(df):

    df = df.copy()

    # Total income of applicant + co-applicant
    df["TotalIncome"] = (
        df["ApplicantIncome"] +
        df["CoapplicantIncome"]
    )

    # Loan amount relative to total income
    df["LoanToIncome"] = (
        df["LoanAmount"] /
        (df["TotalIncome"] + 1)
    )

    return df


# --------------------------------------------------
# SEPARATE FEATURES AND TARGET
# --------------------------------------------------

def split_features_target(df):

    X = df.drop("Loan_Status", axis=1)
    y = df["Loan_Status"].map({
        "Y": 1,
        "N": 0
    })

    return X, y


# --------------------------------------------------
# CREATE PREPROCESSING PIPELINE
# --------------------------------------------------

def create_preprocessor(X):

    numerical_features = X.select_dtypes(
        include=["int64", "float64", "Int64"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object"]
    ).columns.tolist()

    # Numerical preprocessing
    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    # Categorical preprocessing
    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )
        ]
    )

    # Combine both pipelines
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numerical_pipeline,
                numerical_features
            ),
            (
                "cat",
                categorical_pipeline,
                categorical_features
            )
        ]
    )

    return preprocessor


# --------------------------------------------------
# TEST PREPROCESSING
# --------------------------------------------------

if __name__ == "__main__":

    df = load_data()

    print("Original dataset shape:", df.shape)

    # Feature engineering
    df = add_features(df)

    print(
        "Dataset shape after feature engineering:",
        df.shape
    )

    # Split X and y
    X, y = split_features_target(df)

    print("\nFeatures:")
    print(X.columns.tolist())

    print("\nTarget distribution:")
    print(y.value_counts())

    # Create preprocessing pipeline
    preprocessor = create_preprocessor(X)

    print("\nPreprocessing pipeline created successfully!")