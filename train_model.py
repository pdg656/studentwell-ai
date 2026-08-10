import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score
)

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    VotingClassifier
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "data/student_wellness.csv"

MODEL_PATH = "studentwell_model.pkl"
BEST_MODEL_PATH = "studentwell_best_model.pkl"
CONFIG_PATH = "studentwell_model_config.pkl"
IMPORTANCE_PATH = "feature_importance.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.20


# ============================================================
# 1. LOAD DATA
# ============================================================

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Dataset not found at: {DATA_PATH}\n"
        "Run: uv run python generate_data.py"
    )

data = pd.read_csv(DATA_PATH)

print("=" * 70)
print("STUDENTWELL AI - MODEL TRAINING")
print("=" * 70)

print("\nDataset shape:")
print(data.shape)

print("\nRisk distribution:")
print(data["risk_level"].value_counts())


# ============================================================
# 2. DEFINE FEATURES
# ============================================================

# V1:
# Lifestyle and academic features only

FEATURES_V1 = [
    "age",
    "academic_workload",
    "sleep_hours",
    "exercise_days",
    "nutrition_score",
    "study_hours",
    "social_activity"
]


# V2:
# V1 + wellness indicators

FEATURES_V2 = [
    "age",
    "academic_workload",
    "sleep_hours",
    "exercise_days",
    "nutrition_score",
    "study_hours",
    "social_activity",
    "stress_score",
    "anxiety_score",
    "wellness_score"
]

TARGET = "risk_level"


# ============================================================
# 3. VALIDATE DATA
# ============================================================

required_columns = FEATURES_V2 + [TARGET]

missing_columns = [
    column
    for column in required_columns
    if column not in data.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

if data[required_columns].isnull().any().any():
    print("\nWARNING: Missing values detected.")

    print(
        data[required_columns]
        .isnull()
        .sum()
    )

    raise ValueError(
        "Please handle missing values before training."
    )


# ============================================================
# 4. DEFINE X AND Y
# ============================================================

X_v1 = data[FEATURES_V1]
X_v2 = data[FEATURES_V2]

y = data[TARGET]


# ============================================================
# 5. CREATE ONE COMMON TRAIN/TEST SPLIT
# ============================================================

# Create indices first so V1 and V2 use EXACTLY
# the same training and testing rows.

train_indices, test_indices = train_test_split(
    np.arange(len(data)),
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)


X1_train = X_v1.iloc[train_indices]
X1_test = X_v1.iloc[test_indices]

X2_train = X_v2.iloc[train_indices]
X2_test = X_v2.iloc[test_indices]

y_train = y.iloc[train_indices]
y_test = y.iloc[test_indices]


print("\nTraining rows:", len(X1_train))
print("Testing rows:", len(X1_test))


# ============================================================
# 6. CREATE MODELS
# ============================================================

model_v1 = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1
)


model_v2_rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1
)


model_v2_extra = ExtraTreesClassifier(
    n_estimators=500,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1
)


model_v2_hist = HistGradientBoostingClassifier(
    max_iter=300,
    learning_rate=0.05,
    max_leaf_nodes=31,
    l2_regularization=1.0,
    random_state=RANDOM_STATE
)


# ============================================================
# 7. FUNCTION FOR MODEL EVALUATION
# ============================================================

def evaluate_model(
    model,
    X_train,
    X_test,
    y_train,
    y_test,
    model_name
):

    print("\n" + "=" * 70)
    print(f"TRAINING {model_name}")
    print("=" * 70)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    macro_precision = precision_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    macro_recall = recall_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    high_risk_recall = recall_score(
        y_test,
        predictions,
        labels=["High"],
        average="macro",
        zero_division=0
    )

    print("\nRESULTS")

    print(
        f"Accuracy:          {accuracy:.4f}"
    )

    print(
        f"Macro Precision:   {macro_precision:.4f}"
    )

    print(
        f"Macro Recall:      {macro_recall:.4f}"
    )

    print(
        f"Macro F1:          {macro_f1:.4f}"
    )

    print(
        f"High-Risk Recall:  {high_risk_recall:.4f}"
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    print("Confusion Matrix:")

    print(
        confusion_matrix(
            y_test,
            predictions
        )
    )

    return {
        "model": model,
        "predictions": predictions,
        "accuracy": accuracy,
        "precision": macro_precision,
        "recall": macro_recall,
        "f1": macro_f1,
        "high_risk_recall": high_risk_recall
    }


# ============================================================
# 8. TRAIN V1
# ============================================================

v1_result = evaluate_model(
    model_v1,
    X1_train,
    X1_test,
    y_train,
    y_test,
    "V1 RANDOM FOREST"
)


# ============================================================
# 9. TRAIN V2 RANDOM FOREST
# ============================================================

v2_rf_result = evaluate_model(
    model_v2_rf,
    X2_train,
    X2_test,
    y_train,
    y_test,
    "V2 RANDOM FOREST"
)


# ============================================================
# 10. TRAIN V2 EXTRA TREES
# ============================================================

extra_result = evaluate_model(
    model_v2_extra,
    X2_train,
    X2_test,
    y_train,
    y_test,
    "V2 EXTRA TREES"
)


# ============================================================
# 11. TRAIN V2 HISTOGRAM GRADIENT BOOSTING
# ============================================================

hist_result = evaluate_model(
    model_v2_hist,
    X2_train,
    X2_test,
    y_train,
    y_test,
    "V2 HISTOGRAM GRADIENT BOOSTING"
)


# ============================================================
# 12. TRAIN VOTING ENSEMBLE
# ============================================================

print("\n" + "=" * 70)
print("TRAINING V2 ENSEMBLE")
print("=" * 70)


ensemble = VotingClassifier(
    estimators=[
        ("random_forest", model_v2_rf),
        ("extra_trees", model_v2_extra),
        ("hist_gradient", model_v2_hist)
    ],
    voting="soft",
    weights=[3, 3, 2],
    n_jobs=-1
)


ensemble_result = evaluate_model(
    ensemble,
    X2_train,
    X2_test,
    y_train,
    y_test,
    "V2 VOTING ENSEMBLE"
)


# ============================================================
# 13. COMPARE MODELS
# ============================================================

results = pd.DataFrame({

    "Model": [
        "V1 Random Forest",
        "V2 Random Forest",
        "V2 Extra Trees",
        "V2 Hist Gradient Boosting",
        "V2 Ensemble"
    ],

    "Accuracy": [
        v1_result["accuracy"],
        v2_rf_result["accuracy"],
        extra_result["accuracy"],
        hist_result["accuracy"],
        ensemble_result["accuracy"]
    ],

    "Macro Precision": [
        v1_result["precision"],
        v2_rf_result["precision"],
        extra_result["precision"],
        hist_result["precision"],
        ensemble_result["precision"]
    ],

    "Macro Recall": [
        v1_result["recall"],
        v2_rf_result["recall"],
        extra_result["recall"],
        hist_result["recall"],
        ensemble_result["recall"]
    ],

    "Macro F1": [
        v1_result["f1"],
        v2_rf_result["f1"],
        extra_result["f1"],
        hist_result["f1"],
        ensemble_result["f1"]
    ],

    "High Risk Recall": [
        v1_result["high_risk_recall"],
        v2_rf_result["high_risk_recall"],
        extra_result["high_risk_recall"],
        hist_result["high_risk_recall"],
        ensemble_result["high_risk_recall"]
    ]
})


print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(
    results
    .sort_values(
        by="Macro F1",
        ascending=False
    )
    .to_string(index=False)
)


# ============================================================
# 14. SELECT BEST MODEL
# ============================================================

# Macro F1 is the primary selection metric because
# the High-risk class is a minority class.

best_index = results["Macro F1"].idxmax()

best_model_name = results.loc[
    best_index,
    "Model"
]


print("\n" + "=" * 70)
print("BEST MODEL SELECTION")
print("=" * 70)

print(
    f"\nSelected model: {best_model_name}"
)


# ============================================================
# 15. ASSIGN BEST MODEL AND FEATURES
# ============================================================

if best_model_name == "V1 Random Forest":

    best_model = model_v1
    best_features = FEATURES_V1

elif best_model_name == "V2 Random Forest":

    best_model = model_v2_rf
    best_features = FEATURES_V2

elif best_model_name == "V2 Extra Trees":

    best_model = model_v2_extra
    best_features = FEATURES_V2

elif best_model_name == "V2 Hist Gradient Boosting":

    best_model = model_v2_hist
    best_features = FEATURES_V2

elif best_model_name == "V2 Ensemble":

    best_model = ensemble
    best_features = FEATURES_V2

else:

    raise ValueError(
        f"Unknown model: {best_model_name}"
    )


# ============================================================
# 16. SAVE BEST MODEL
# ============================================================

# Main filename for your Streamlit application

joblib.dump(
    best_model,
    MODEL_PATH
)


# Backup / descriptive filename

joblib.dump(
    best_model,
    BEST_MODEL_PATH
)


# ============================================================
# 17. SAVE MODEL CONFIGURATION
# ============================================================

model_config = {
    "features": best_features,
    "target": TARGET,
    "model_name": best_model_name,
    "random_state": RANDOM_STATE
}


joblib.dump(
    model_config,
    CONFIG_PATH
)


print("\n" + "=" * 70)
print("MODEL SAVED")
print("=" * 70)

print(
    f"\nMain model:"
    f"\n{MODEL_PATH}"
)

print(
    f"\nBackup model:"
    f"\n{BEST_MODEL_PATH}"
)

print(
    f"\nConfiguration:"
    f"\n{CONFIG_PATH}"
)

print(
    f"\nBest model:"
    f"\n{best_model_name}"
)

print(
    "\nFeatures used:"
)

print(best_features)


# ============================================================
# 18. FEATURE IMPORTANCE
# ============================================================

# Random Forest and Extra Trees provide
# feature_importances_.

if hasattr(best_model, "feature_importances_"):

    importance = pd.DataFrame({
        "feature": best_features,
        "importance": best_model.feature_importances_
    })

    importance = importance.sort_values(
        by="importance",
        ascending=False
    )

    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE")
    print("=" * 70)

    print(
        importance.to_string(index=False)
    )

    importance.to_csv(
        IMPORTANCE_PATH,
        index=False
    )

    print(
        f"\nFeature importance saved to:"
        f"\n{IMPORTANCE_PATH}"
    )

else:

    print(
        "\nFeature importance is not directly "
        "available for the selected model."
    )


# ============================================================
# 19. CROSS-VALIDATION CHECK
# ============================================================

print("\n" + "=" * 70)
print("5-FOLD CROSS-VALIDATION")
print("=" * 70)

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)


# Cross-validation is performed on the training data
# to avoid using the final test set.

try:

    cv_scores = cross_val_score(
        best_model,
        X2_train if best_features == FEATURES_V2 else X1_train,
        y_train,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1
    )

    print(
        "\nCross-validation Macro F1 scores:"
    )

    print(
        np.round(cv_scores, 4)
    )

    print(
        f"\nMean CV Macro F1: "
        f"{cv_scores.mean():.4f}"
    )

    print(
        f"CV Standard Deviation: "
        f"{cv_scores.std():.4f}"
    )

except Exception as e:

    print(
        "\nCross-validation warning:"
    )

    print(e)


# ============================================================
# 20. FINAL SUMMARY
# ============================================================

best_row = results.loc[best_index]

print("\n" + "=" * 70)
print("FINAL MODEL SUMMARY")
print("=" * 70)

print(
    f"\nBest Model:          {best_model_name}"
)

print(
    f"Test Accuracy:       {best_row['Accuracy']:.4f}"
)

print(
    f"Test Macro Precision:{best_row['Macro Precision']:.4f}"
)

print(
    f"Test Macro Recall:   {best_row['Macro Recall']:.4f}"
)

print(
    f"Test Macro F1:       {best_row['Macro F1']:.4f}"
)

print(
    f"High-Risk Recall:    {best_row['High Risk Recall']:.4f}"
)

print(
    "\nModel files:"
)

print(
    f"  {MODEL_PATH}"
)

print(
    f"  {BEST_MODEL_PATH}"
)

print(
    f"  {CONFIG_PATH}"
)

print("\nTraining complete!")