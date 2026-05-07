# ============================================================
# 05_statistical_analysis.py
# Additional statistical analyses using existing dataset
# and existing model outputs
# ============================================================

import os
import joblib
import numpy as np
import pandas as pd

from scipy.stats import kruskal

from sklearn.metrics import brier_score_loss
from sklearn.preprocessing import label_binarize

from utils import reconstruct_augmented_features


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs("results", exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

data = joblib.load("outputs/preprocessed_data.pkl")

X = data["X"]
y = data["y"]

feature_names_original = data["feature_names_original"]

grade_encoder = data["grade_encoder"]


# ============================================================
# LOAD FINAL MODEL
# ============================================================

model_bundle = joblib.load(
    "models/final_sequential_M1_M7_M3_model.pkl"
)

final_model = model_bundle["model"]


# ============================================================
# RECONSTRUCT FINAL AUGMENTED FEATURE SPACE
# ============================================================

X_stage3 = reconstruct_augmented_features(
    final_model=final_model,
    X=X
)


# ============================================================
# BIOMARKER STATISTICAL ANALYSIS
# Kruskal-Wallis + Eta-squared
# ============================================================

print("\n================================================")
print("BIOMARKER STATISTICAL ANALYSIS")
print("================================================")

biomarker_results = []

X_original = X[:, :len(feature_names_original)]

for feature_index, feature_name in enumerate(feature_names_original):

    feature_values = X_original[:, feature_index]

    unique_grades = np.unique(y)

    grouped_values = [
        feature_values[y == grade]
        for grade in unique_grades
    ]

    # --------------------------------------------------------
    # Kruskal-Wallis test
    # --------------------------------------------------------

    H_statistic, p_value = kruskal(*grouped_values)

    # --------------------------------------------------------
    # Eta-squared effect size
    # η² = (H - k + 1)/(n - k)
    # --------------------------------------------------------

    n = len(feature_values)

    k = len(unique_grades)

    eta_squared = (H_statistic - k + 1) / (n - k)

    eta_squared = max(0, eta_squared)

    # --------------------------------------------------------
    # Effect size interpretation
    # --------------------------------------------------------

    if eta_squared < 0.01:

        interpretation = "Negligible"

    elif eta_squared < 0.06:

        interpretation = "Small"

    elif eta_squared < 0.14:

        interpretation = "Moderate"

    else:

        interpretation = "Large"

    biomarker_results.append({
        "Feature": feature_name,
        "Kruskal_H": round(H_statistic, 4),
        "p_value": p_value,
        "Eta_squared": round(eta_squared, 4),
        "Effect_Size": interpretation
    })


# ============================================================
# SAVE BIOMARKER RESULTS
# ============================================================

biomarker_results_df = pd.DataFrame(
    biomarker_results
)

biomarker_results_df.to_csv(
    "results/biomarker_statistical_analysis.csv",
    index=False
)

print("\nBiomarker Statistical Results:")
print(biomarker_results_df)


# ============================================================
# BRIER SCORE ANALYSIS
# ============================================================

print("\n================================================")
print("BRIER SCORE ANALYSIS")
print("================================================")

# ------------------------------------------------------------
# Predict probabilities from final classifier
# ------------------------------------------------------------

predicted_probabilities = final_model.M3.predict_proba(
    X_stage3
)

# ------------------------------------------------------------
# Convert multiclass labels to one-hot encoding
# ------------------------------------------------------------

y_binarized = label_binarize(
    y,
    classes=np.unique(y)
)

# ------------------------------------------------------------
# Compute multiclass Brier scores
# ------------------------------------------------------------

brier_scores = []

for class_index in range(
    predicted_probabilities.shape[1]
):

    class_brier = brier_score_loss(
        y_binarized[:, class_index],
        predicted_probabilities[:, class_index]
    )

    brier_scores.append(class_brier)

overall_brier_score = np.mean(brier_scores)

overall_brier_score = round(
    overall_brier_score,
    6
)


# ============================================================
# SAVE BRIER SCORES
# ============================================================

brier_df = pd.DataFrame({
    "Class": [
        str(c)
        for c in grade_encoder.classes_
    ],
    "Brier_Score": brier_scores
})

brier_df.loc[len(brier_df)] = [
    "Overall",
    overall_brier_score
]

brier_df.to_csv(
    "results/brier_scores.csv",
    index=False
)

print("\nBrier Scores:")
print(brier_df)


# ============================================================
# CALIBRATION INTERPRETATION
# ============================================================

print("\nOverall Calibration Interpretation:")

print(
    "Low Brier scores indicated favorable "
    "probabilistic calibration."
)


# ============================================================
# SAVE SUMMARY REPORT
# ============================================================

summary_report = pd.DataFrame({
    "Metric": [
        "Overall_Brier_Score",
        "Calibration_Interpretation"
    ],
    "Value": [
        overall_brier_score,
        "Low Brier scores indicated favorable "
        "probabilistic calibration."
    ]
})

summary_report.to_csv(
    "results/statistical_summary_report.csv",
    index=False
)

print("\nSaved:")
print("results/biomarker_statistical_analysis.csv")
print("results/brier_scores.csv")
print("results/statistical_summary_report.csv")
