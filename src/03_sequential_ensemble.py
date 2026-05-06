import os
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report

from utils import (
    get_outer_cv,
    get_inner_cv,
    get_M1,
    get_M7,
    get_M3,
    leakage_safe_oof_predictions,
    fit_with_smote,
    compute_metrics,
    SequentialM1M7M3DeploymentModel
)


os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

data = joblib.load("outputs/preprocessed_data.pkl")

X = data["X"]
y = data["y"]
feature_names_original = data["feature_names_original"]
sex_encoder = data["sex_encoder"]
grade_encoder = data["grade_encoder"]

outer_cv = get_outer_cv()
inner_cv = get_inner_cv()


def run_sequential_ensemble(X, y):

    all_true = []
    all_pred = []
    fold_records = []

    for fold, (train_idx, test_idx) in enumerate(
        outer_cv.split(X, y),
        start=1
    ):

        print(f"\n================ Fold {fold} ================")

        X_train_raw = X[train_idx]
        X_test_raw = X[test_idx]

        y_train = y[train_idx]
        y_test = y[test_idx]

        scaler = StandardScaler()

        X_train = scaler.fit_transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)

        M1 = get_M1()

        pred1_train = leakage_safe_oof_predictions(
            model=M1,
            X_train=X_train,
            y_train=y_train,
            inner_cv=inner_cv
        )

        M1_full = fit_with_smote(
            model=M1,
            X_train=X_train,
            y_train=y_train
        )

        pred1_test = M1_full.predict(X_test)
        pred1_test = np.asarray(pred1_test).ravel()

        X_train_stage2 = np.column_stack([
            X_train,
            pred1_train
        ])

        X_test_stage2 = np.column_stack([
            X_test,
            pred1_test
        ])

        M7 = get_M7()

        pred2_train = leakage_safe_oof_predictions(
            model=M7,
            X_train=X_train_stage2,
            y_train=y_train,
            inner_cv=inner_cv
        )

        M7_full = fit_with_smote(
            model=M7,
            X_train=X_train_stage2,
            y_train=y_train
        )

        pred2_test = M7_full.predict(X_test_stage2)
        pred2_test = np.asarray(pred2_test).ravel()

        X_train_stage3 = np.column_stack([
            X_train_stage2,
            pred2_train
        ])

        X_test_stage3 = np.column_stack([
            X_test_stage2,
            pred2_test
        ])

        M3 = get_M3()

        M3_full = fit_with_smote(
            model=M3,
            X_train=X_train_stage3,
            y_train=y_train
        )

        final_pred = M3_full.predict(X_test_stage3)
        final_pred = np.asarray(final_pred).ravel()

        fold_metrics = compute_metrics(
            y_true=y_test,
            y_pred=final_pred
        )

        fold_metrics["Fold"] = fold

        print(fold_metrics)

        fold_records.append(fold_metrics)

        all_true.extend(y_test)
        all_pred.extend(final_pred)

    return (
        np.array(all_true),
        np.array(all_pred),
        pd.DataFrame(fold_records)
    )


y_true, y_pred, fold_results = run_sequential_ensemble(X, y)

overall_metrics = compute_metrics(y_true, y_pred)

fold_results.to_csv(
    "outputs/fold_wise_results.csv",
    index=False
)

pd.DataFrame([overall_metrics]).to_csv(
    "outputs/overall_metrics.csv",
    index=False
)

cm = confusion_matrix(y_true, y_pred)

pd.DataFrame(cm).to_csv(
    "outputs/confusion_matrix.csv",
    index=False
)

report = classification_report(
    y_true,
    y_pred,
    target_names=[str(c) for c in grade_encoder.classes_],
    zero_division=0,
    output_dict=True
)

pd.DataFrame(report).transpose().to_csv(
    "outputs/classification_report.csv"
)

print("\nOverall metrics:")
print(overall_metrics)

print("\nConfusion matrix:")
print(cm)

final_model = SequentialM1M7M3DeploymentModel()
final_model.fit(X, y)

joblib.dump(
    {
        "model": final_model,
        "sex_encoder": sex_encoder,
        "grade_encoder": grade_encoder,
        "feature_names_original": feature_names_original
    },
    "models/final_sequential_M1_M7_M3_model.pkl"
)

print("\nSaved final model:")
print("models/final_sequential_M1_M7_M3_model.pkl")
