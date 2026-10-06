import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split

from preprocessing import (
    load_data,
    add_features,
    split_features_target
)


# ============================================================
# 1. LOAD DATASET
# ============================================================

df = load_data()

df = add_features(df)

X, y = split_features_target(df)


# ============================================================
# 2. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ============================================================
# 3. LOAD TUNED RANDOM FOREST
# ============================================================

model_path = "models/random_forest_tuned.pkl"

model = joblib.load(model_path)

print("Tuned Random Forest loaded successfully.")


# ============================================================
# 4. GET PREPROCESSOR AND CLASSIFIER
# ============================================================

preprocessor = model.named_steps["preprocessor"]

classifier = model.named_steps["model"]


# ============================================================
# 5. TRANSFORM TEST DATA
# ============================================================

X_test_transformed = preprocessor.transform(X_test)

feature_names = preprocessor.get_feature_names_out()

print(
    "Number of transformed features:",
    len(feature_names)
)


# ============================================================
# 6. CREATE SHAP EXPLAINER
# ============================================================

explainer = shap.TreeExplainer(classifier)

shap_values = explainer.shap_values(
    X_test_transformed
)


# ============================================================
# 7. HANDLE DIFFERENT SHAP OUTPUT FORMATS
# ============================================================

if isinstance(shap_values, list):

    # Older SHAP versions
    shap_values_positive = shap_values[1]

elif hasattr(shap_values, "shape") and len(shap_values.shape) == 3:

    # Newer SHAP versions
    # Shape:
    # (samples, features, classes)

    shap_values_positive = shap_values[:, :, 1]

else:

    # Already:
    # (samples, features)

    shap_values_positive = shap_values


# ============================================================
# 8. VERIFY SHAP SHAPE
# ============================================================

print(
    "SHAP values shape:",
    shap_values_positive.shape
)

print(
    "Expected feature count:",
    len(feature_names)
)


# ============================================================
# 9. CREATE OUTPUT DIRECTORY
# ============================================================

output_dir = "data/shap_results"

os.makedirs(
    output_dir,
    exist_ok=True
)


# ============================================================
# 10. GLOBAL SHAP FEATURE IMPORTANCE
# ============================================================

importance = abs(
    shap_values_positive
).mean(axis=0)


# Make sure importance is one-dimensional
importance = importance.ravel()


# ============================================================
# 11. CREATE FEATURE IMPORTANCE DATAFRAME
# ============================================================

feature_importance = pd.DataFrame({
    "Feature": feature_names,
    "SHAP Importance": importance
})


# Sort from highest to lowest
feature_importance = feature_importance.sort_values(
    "SHAP Importance",
    ascending=False
)


# ============================================================
# 12. DISPLAY TOP FEATURES
# ============================================================

print("\n" + "=" * 60)
print("TOP SHAP FEATURES")
print("=" * 60)

print(
    feature_importance.head(15).to_string(
        index=False
    )
)


# ============================================================
# 13. SAVE FEATURE IMPORTANCE CSV
# ============================================================

feature_importance.to_csv(
    os.path.join(
        output_dir,
        "shap_feature_importance.csv"
    ),
    index=False
)


# ============================================================
# 14. SHAP BAR PLOT
# ============================================================

plt.figure(figsize=(10, 7))

shap.summary_plot(
    shap_values_positive,
    X_test_transformed,
    feature_names=feature_names,
    plot_type="bar",
    show=False
)

plt.title(
    "SHAP Global Feature Importance"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_dir,
        "shap_feature_importance.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 15. SHAP SUMMARY PLOT
# ============================================================

plt.figure(figsize=(10, 7))

shap.summary_plot(
    shap_values_positive,
    X_test_transformed,
    feature_names=feature_names,
    show=False
)

plt.title(
    "SHAP Feature Impact Summary"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_dir,
        "shap_summary.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 16. COMPLETION MESSAGE
# ============================================================

print("\n" + "=" * 60)
print("SHAP ANALYSIS COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nGenerated files:")

print(
    "1. data/shap_results/shap_feature_importance.png"
)

print(
    "2. data/shap_results/shap_summary.png"
)

print(
    "3. data/shap_results/shap_feature_importance.csv"
)