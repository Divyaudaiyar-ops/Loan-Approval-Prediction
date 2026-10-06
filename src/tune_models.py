import pandas as pd
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from preprocessing import (
    load_data,
    add_features,
    split_features_target,
    create_preprocessor
)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

df = load_data()

df = add_features(df)

X, y = split_features_target(df)


# --------------------------------------------------
# TRAIN-TEST SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# --------------------------------------------------
# PREPROCESSOR
# --------------------------------------------------

preprocessor = create_preprocessor(X_train)


# --------------------------------------------------
# CROSS VALIDATION
# --------------------------------------------------

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# --------------------------------------------------
# RANDOM FOREST
# --------------------------------------------------

rf_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(
        random_state=42
    ))
])

rf_params = {
    "model__n_estimators": [100, 200, 300],
    "model__max_depth": [None, 5, 8, 10, 15],
    "model__min_samples_split": [2, 5, 10],
    "model__min_samples_leaf": [1, 2, 4],
    "model__max_features": ["sqrt", "log2"]
}


# --------------------------------------------------
# SVM
# --------------------------------------------------

svm_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", SVC(
        probability=True,
        random_state=42
    ))
])

svm_params = {
    "model__C": [0.1, 1, 10, 100],
    "model__gamma": ["scale", "auto", 0.01, 0.1],
    "model__kernel": ["rbf", "linear"]
}


# --------------------------------------------------
# XGBOOST
# --------------------------------------------------

xgb_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", XGBClassifier(
        eval_metric="logloss",
        random_state=42
    ))
])

xgb_params = {
    "model__n_estimators": [100, 200, 300],
    "model__max_depth": [2, 3, 4, 5, 6],
    "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
    "model__subsample": [0.7, 0.8, 1.0],
    "model__colsample_bytree": [0.7, 0.8, 1.0]
}


# --------------------------------------------------
# TUNING FUNCTION
# --------------------------------------------------

def tune_model(name, pipeline, parameters):

    print("\n" + "=" * 60)
    print(f"TUNING {name}")
    print("=" * 60)

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=parameters,
        n_iter=15,
        scoring="roc_auc",
        cv=cv,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    search.fit(X_train, y_train)

    print("\nBest Parameters:")
    print(search.best_params_)

    print(
        f"\nBest CV ROC-AUC: "
        f"{search.best_score_:.4f}"
    )

    filename = (
        name.lower()
        .replace(" ", "_")
        + "_tuned.pkl"
    )

    joblib.dump(
        search.best_estimator_,
        Path("models") / filename
    )

    print(f"Saved: models/{filename}")

    return search


# --------------------------------------------------
# RUN TUNING
# --------------------------------------------------

rf_search = tune_model(
    "Random Forest",
    rf_pipeline,
    rf_params
)

svm_search = tune_model(
    "SVM",
    svm_pipeline,
    svm_params
)

xgb_search = tune_model(
    "XGBoost",
    xgb_pipeline,
    xgb_params
)


print("\n" + "=" * 60)
print("HYPERPARAMETER TUNING COMPLETED")
print("=" * 60)