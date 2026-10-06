import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
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

# Feature engineering
df = add_features(df)

# Separate features and target
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
# PREPROCESSING
# --------------------------------------------------

preprocessor = create_preprocessor(X_train)


# --------------------------------------------------
# DEFINE MODELS
# --------------------------------------------------

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=5,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        random_state=42
    ),

    "SVM": SVC(
        probability=True,
        random_state=42
    ),

    "XGBoost": XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42
    )
}


# --------------------------------------------------
# CREATE MODEL DIRECTORY
# --------------------------------------------------

model_dir = Path("models")
model_dir.mkdir(exist_ok=True)


# --------------------------------------------------
# CROSS-VALIDATION
# --------------------------------------------------

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


results = []


# --------------------------------------------------
# TRAIN MODELS
# --------------------------------------------------

print("=" * 60)
print("MODEL TRAINING")
print("=" * 60)

for name, model in models.items():

    print(f"\nTraining {name}...")

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    # Cross-validation
    scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring="accuracy"
    )

    # Train on complete training data
    pipeline.fit(X_train, y_train)

    # Save model
    filename = name.lower().replace(" ", "_") + ".pkl"

    joblib.dump(
        pipeline,
        model_dir / filename
    )

    results.append({
        "Model": name,
        "CV Mean Accuracy": scores.mean(),
        "CV Std": scores.std()
    })

    print(
        f"Mean CV Accuracy: {scores.mean():.4f}"
    )

    print(
        f"CV Standard Deviation: {scores.std():.4f}"
    )

    print(
        f"Saved: models/{filename}"
    )


# --------------------------------------------------
# SAVE RESULTS
# --------------------------------------------------

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "CV Mean Accuracy",
    ascending=False
)

results_df.to_csv(
    "model_results.csv",
    index=False
)


# --------------------------------------------------
# DISPLAY RESULTS
# --------------------------------------------------

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(results_df.to_string(index=False))

print("\nAll models trained successfully!")