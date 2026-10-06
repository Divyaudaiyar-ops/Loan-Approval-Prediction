import pandas as pd
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from preprocessing import (
    load_data,
    add_features,
    split_features_target
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
# MODEL FILES
# --------------------------------------------------

model_files = {
    "Logistic Regression": "models/logistic_regression.pkl",
    "Decision Tree": "models/decision_tree.pkl",
    "Random Forest": "models/random_forest.pkl",
    "SVM": "models/svm.pkl",
    "XGBoost": "models/xgboost.pkl"
}


results = []


# --------------------------------------------------
# EVALUATE EACH MODEL
# --------------------------------------------------

for name, file_path in model_files.items():

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    model = joblib.load(file_path)

    # Predictions
    y_pred = model.predict(X_test)

    # Probabilities
    y_probability = model.predict_proba(X_test)[:, 1]

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        y_probability
    )

    # Confusion matrix
    cm = confusion_matrix(
        y_test,
        y_pred
    )

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["Rejected", "Approved"],
            zero_division=0
        )
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": roc_auc
    })


# --------------------------------------------------
# SAVE RESULTS
# --------------------------------------------------

results_df = pd.DataFrame(results)

results_df.to_csv(
    "evaluation_results.csv",
    index=False
)


print("\n" + "=" * 60)
print("FINAL MODEL EVALUATION")
print("=" * 60)

print(
    results_df.to_string(index=False)
)

print("\nEvaluation completed successfully!")