import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from scipy.stats import zscore

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.base import clone
from sklearn.ensemble import BaggingClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier

from imblearn.over_sampling import BorderlineSMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from catboost import CatBoostClassifier


# ============================================================
# GLOBAL SETTINGS
# ============================================================

RANDOM_STATE = 42

REQUIRED_COLUMNS = [
    "Age",
    "Sex",
    "Beta_HCG",
    "AFP",
    "PD-L1",
    "Grade"
]

NUMERIC_COLUMNS = [
    "Age",
    "Beta_HCG",
    "AFP",
    "PD-L1"
]


# ============================================================
# DATA LOADING AND PREPROCESSING
# ============================================================

def load_and_preprocess_data(data_path):
    df = pd.read_csv(data_path)

    redundant_columns = [
        col for col in df.columns
        if col not in REQUIRED_COLUMNS
    ]

    df = df[REQUIRED_COLUMNS].copy()

    original_shape = df.shape

    # Remove duplicates and missing/null samples
    df = df.drop_duplicates()
    df = df.dropna()

    # Remove statistical outliers using z-score threshold
    z_scores = np.abs(zscore(df[NUMERIC_COLUMNS]))
    df = df[(z_scores < 3).all(axis=1)].copy()

    # Encode Sex as a single binary feature
    sex_encoder = LabelEncoder()
    df["Sex"] = sex_encoder.fit_transform(df["Sex"])

    # Encode target labels
    grade_encoder = LabelEncoder()
    df["Grade"] = grade_encoder.fit_transform(df["Grade"])

    X_df = df.drop(columns=["Grade"])
    y = df["Grade"].values
    X = X_df.values

    feature_names_original = list(X_df.columns)

    cleaning_report = {
        "original_shape": original_shape,
        "final_shape": df.shape,
        "redundant_columns_removed": redundant_columns,
        "features": feature_names_original,
        "sex_encoding": dict(
            zip(
                sex_encoder.classes_,
                sex_encoder.transform(sex_encoder.classes_)
            )
        )
    }

    return (
        X,
        y,
        feature_names_original,
        sex_encoder,
        grade_encoder,
        cleaning_report
    )


# ============================================================
# CROSS-VALIDATION
# ============================================================

def get_outer_cv():
    return StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE
    )


def get_inner_cv():
    return StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE
    )


# ============================================================
# MODEL DEFINITIONS FROM TABLE S2
# M1 = Bagging
# M7 = CatBoost
# M3 = ExtraTrees
# ============================================================

def get_M1():
    return BaggingClassifier(
        estimator=DecisionTreeClassifier(random_state=RANDOM_STATE),
        bootstrap=True,
        max_samples=16,
        n_estimators=78,
        max_features=4,
        random_state=RANDOM_STATE
    )


def get_M7():
    return CatBoostClassifier(
        depth=5,
        learning_rate=0.69739829,
        iterations=80,
        loss_function="MultiClass",
        verbose=0,
        random_seed=RANDOM_STATE
    )


def get_M3():
    return ExtraTreesClassifier(
        min_weight_fraction_leaf=0.0128358,
        bootstrap=True,
        max_depth=39,
        max_leaf_nodes=39,
        n_estimators=57,
        max_features=4,
        criterion="gini",
        min_samples_split=12,
        min_samples_leaf=2,
        random_state=RANDOM_STATE
    )


# ============================================================
# LEAKAGE-SAFE OOF PREDICTIONS
# SMOTE is applied inside each inner training fold only
# ============================================================

def leakage_safe_oof_predictions(model, X_train, y_train, inner_cv):
    pipe = ImbPipeline([
        (
            "smote",
            BorderlineSMOTE(
                random_state=RANDOM_STATE,
                k_neighbors=2
            )
        ),
        ("model", clone(model))
    ])

    oof_pred = cross_val_predict(
        pipe,
        X_train,
        y_train,
        cv=inner_cv,
        method="predict",
        n_jobs=-1
    )

    return np.asarray(oof_pred).ravel()


# ============================================================
# FIT MODEL WITH SMOTE ON TRAINING DATA ONLY
# ============================================================

def fit_with_smote(model, X_train, y_train):
    smote = BorderlineSMOTE(
        random_state=RANDOM_STATE,
        k_neighbors=2
    )

    X_balanced, y_balanced = smote.fit_resample(
        X_train,
        y_train
    )

    fitted_model = clone(model)
    fitted_model.fit(X_balanced, y_balanced)

    return fitted_model


# ============================================================
# RECONSTRUCT AUGMENTED FEATURE SPACE
# Original features → M1 prediction → M7 prediction
# ============================================================

def reconstruct_augmented_features(final_model, X):
    X_scaled = final_model.scaler.transform(X)

    pred1 = final_model.M1.predict(X_scaled)
    pred1 = np.asarray(pred1).ravel()

    X_stage2 = np.column_stack([
        X_scaled,
        pred1
    ])

    pred2 = final_model.M7.predict(X_stage2)
    pred2 = np.asarray(pred2).ravel()

    X_stage3 = np.column_stack([
        X_stage2,
        pred2
    ])

    return X_stage3


# ============================================================
# FINAL DEPLOYMENT MODEL
# Trained on full data after evaluation
# ============================================================

class SequentialM1M7M3DeploymentModel:

    def __init__(self):
        self.scaler = StandardScaler()
        self.M1 = get_M1()
        self.M7 = get_M7()
        self.M3 = get_M3()

    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)

        self.M1 = fit_with_smote(
            self.M1,
            X_scaled,
            y
        )

        pred1 = self.M1.predict(X_scaled)
        pred1 = np.asarray(pred1).ravel()

        X_stage2 = np.column_stack([
            X_scaled,
            pred1
        ])

        self.M7 = fit_with_smote(
            self.M7,
            X_stage2,
            y
        )

        pred2 = self.M7.predict(X_stage2)
        pred2 = np.asarray(pred2).ravel()

        X_stage3 = np.column_stack([
            X_stage2,
            pred2
        ])

        self.M3 = fit_with_smote(
            self.M3,
            X_stage3,
            y
        )

        return self

    def predict(self, X):
        X_stage3 = reconstruct_augmented_features(
            self,
            X
        )

        pred = self.M3.predict(X_stage3)

        return np.asarray(pred).ravel()

    def predict_proba(self, X):
        X_stage3 = reconstruct_augmented_features(
            self,
            X
        )

        return self.M3.predict_proba(X_stage3)


# ============================================================
# METRIC COMPUTATION
# ============================================================

def compute_metrics(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(
            y_true,
            y_pred
        ),
        "Precision": precision_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        ),
        "Recall": recall_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        ),
        "F1": f1_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        )
    }
