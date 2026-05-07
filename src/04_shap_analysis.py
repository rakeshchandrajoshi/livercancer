# # ============================================================
# # SHAP EXPLAINABILITY
# # SHAP values are computed using the final ensemble classifier.
# # For interpretability, SHAP visualizations are restricted to
# # the original biomarker and demographic features:
# # Age, Sex, Beta_HCG, AFP, and PD-L1.
# # ============================================================

import shap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# # ------------------------------------------------------------
# # Reconstruct augmented ensemble feature space
# # exactly as used by final M3 classifier
# # ------------------------------------------------------------

X_scaled = final_model.scaler.transform(X)

# Stage 1: M1 prediction
pred1 = final_model.M1.predict(X_scaled)
pred1 = np.asarray(pred1).ravel()

X_stage2 = np.column_stack([
    X_scaled,
    pred1
])

# Stage 2: M7 prediction
pred2 = final_model.M7.predict(X_stage2)
pred2 = np.asarray(pred2).ravel()

X_stage3 = np.column_stack([
    X_stage2,
    pred2
])



# ============================================================
# SHAP EXPLAINABILITY
# ============================================================

explainer = shap.TreeExplainer(final_model.M3)

shap_values_raw = explainer.shap_values(X_stage3)

n_original_features = len(feature_names_original)

X_visualization = X_stage3[:, :n_original_features]

# ------------------------------------------------------------
# Convert SHAP output into multiclass-safe list format
# Each element must be: n_samples × n_original_features
# ------------------------------------------------------------

if isinstance(shap_values_raw, list):

    shap_values_visualization = [
        sv[:, :n_original_features]
        for sv in shap_values_raw
    ]

else:

    shap_values_raw = np.asarray(shap_values_raw)

    # New SHAP format:
    # n_samples × n_features × n_classes
    if shap_values_raw.ndim == 3 and shap_values_raw.shape[1] == X_stage3.shape[1]:

        shap_values_visualization = [
            shap_values_raw[:, :n_original_features, class_idx]
            for class_idx in range(shap_values_raw.shape[2])
        ]

    # Alternative SHAP format:
    # n_classes × n_samples × n_features
    elif shap_values_raw.ndim == 3 and shap_values_raw.shape[2] == X_stage3.shape[1]:

        shap_values_visualization = [
            shap_values_raw[class_idx, :, :n_original_features]
            for class_idx in range(shap_values_raw.shape[0])
        ]

    # Binary / single-output format
    elif shap_values_raw.ndim == 2:

        shap_values_visualization = shap_values_raw[:, :n_original_features]

    else:
        raise ValueError(
            f"Unexpected SHAP shape: {shap_values_raw.shape}"
        )


# ------------------------------------------------------------
# Debug shape check
# ------------------------------------------------------------

print("X_visualization shape:", X_visualization.shape)

if isinstance(shap_values_visualization, list):
    for i, sv in enumerate(shap_values_visualization):
        print(f"SHAP class {i} shape:", sv.shape)
else:
    print("SHAP shape:", shap_values_visualization.shape)


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
    "SHAP_global_feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# SHAP SUMMARY DOT PLOT
# For multiclass, plot class-wise summary plots
# ============================================================

if isinstance(shap_values_visualization, list):

    for class_idx, sv in enumerate(shap_values_visualization):

        shap.summary_plot(
            sv,
            X_visualization,
            feature_names=feature_names_original,
            show=False
        )

        plt.savefig(
            f"SHAP_summary_dot_plot_class_{class_idx}.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

else:

    shap.summary_plot(
        shap_values_visualization,
        X_visualization,
        feature_names=feature_names_original,
        show=False
    )

    plt.savefig(
        "SHAP_summary_dot_plot.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# MEAN SHAP VALUE BY GRADE
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

    for class_idx, class_label in enumerate(grade_labels):

        class_shap = shap_values_visualization[class_idx]

        mean_shap_by_grade.loc[class_label, :] = np.mean(
            np.abs(class_shap),
            axis=0
        )

else:

    mean_shap_by_grade.loc[grade_labels[0], :] = np.mean(
        np.abs(shap_values_visualization),
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
    "SHAP_mean_by_grade.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved SHAP plots successfully.")
