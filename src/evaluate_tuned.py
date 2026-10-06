import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
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
# SAME TRAIN-TEST SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# --------------------------------------------------
# TUNED MODELS
# --------------------------------------------------

models = {
    "Random Forest Tuned":
        "models/random_forest_tuned.pkl",

    "SVM Tuned":
        "models/svm_tuned.pkl",

    "XGBoost Tuned":
        "models/xgboost_tuned.pkl"
}


results = []


# --------------------------------------------------
# EVALUATION
# --------------------------------------------------

for name, path in models.items():

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    model = joblib.load(path)

    y_pred = model.predict(X_test)

    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

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
        y_prob
    )

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
    "tuned_model_results.csv",
    index=False
)

print("\n" + "=" * 60)
print("TUNED MODEL COMPARISON")
print("=" * 60)

print(
    results_df.to_string(index=False)
)

print("\nTuned model evaluation completed!")