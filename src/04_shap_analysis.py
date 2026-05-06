import os
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from utils import reconstruct_augmented_features


os.makedirs("results", exist_ok=True)

data = joblib.load("outputs/preprocessed_data.pkl")
model_bundle = joblib.load("models/final_sequential_M1_M7_M3_model.pkl")

X = data["X"]
feature_names_original = model_bundle["feature_names_original"]
grade_encoder = model_bundle["grade_encoder"]
final_model = model_bundle["model"]

X_stage3 = reconstruct_augmented_features(
    final_model=final_model,
    X=X
)

explainer = shap.TreeExplainer(final_model.M3)

shap_values = explainer.shap_values(X_stage3)

n_original_features = len(feature_names_original)

X_visualization = X_stage3[:, :n_original_features]

if isinstance(shap_values, list):
    shap_values_visualization = [
        sv[:, :n_original_features]
        for sv in shap_values
    ]
else:
    shap_values_visualization = shap_values[:, :, :n_original_features]


# ============================================================
# SHAP GLOBAL FEATURE IMPORTANCE BAR PLOT
# ============================================================

shap.summary_plot(
    shap_values_visualization,
    X_visualization,
    feature_names=feature_names_original,
    plot_type="bar",
    show=False
)

plt.savefig(
    "results/SHAP_global_feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# SHAP MEAN VALUE BY GRADE
# ============================================================

grade_labels = [
    str(label)
    for label in grade_encoder.classes_
]

mean_shap_by_grade = pd.DataFrame(
    index=grade_labels,
    columns=feature_names_original
)

if isinstance(shap_values_visualization, list):

    for class_index, class_label in enumerate(grade_labels):

        class_shap = shap_values_visualization[class_index]

        mean_shap_by_grade.loc[class_label, :] = np.mean(
            np.abs(class_shap),
            axis=0
        )

else:

    for class_index, class_label in enumerate(grade_labels):

        class_shap = shap_values_visualization[:, :, class_index]

        mean_shap_by_grade.loc[class_label, :] = np.mean(
            np.abs(class_shap),
            axis=0
        )

mean_shap_by_grade = mean_shap_by_grade.astype(float)

display_grade_labels = []

for label in grade_labels:
    if str(label) == "0":
        display_grade_labels.append("Control")
    else:
        display_grade_labels.append(f"Grade {label}")

mean_shap_by_grade.index = display_grade_labels

mean_shap_by_grade.to_csv(
    "results/SHAP_mean_by_grade.csv"
)

plt.figure(figsize=(9, 5))

for feature in feature_names_original:
    plt.plot(
        mean_shap_by_grade.index,
        mean_shap_by_grade[feature],
        marker="o",
        label=feature
    )

plt.xlabel("Grade")
plt.ylabel("Mean SHAP Value")
plt.xticks(rotation=45)
plt.legend(title="Features")
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    "results/SHAP_mean_by_grade.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# SHAP SUMMARY DOT PLOT
# ============================================================

shap.summary_plot(
    shap_values_visualization,
    X_visualization,
    feature_names=feature_names_original,
    show=False
)

plt.savefig(
    "results/SHAP_summary_dot_plot.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved SHAP outputs in results/")
