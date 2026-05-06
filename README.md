# Liver Cancer Grading Using a Leakage-Safe Sequential Ensemble Framework

## Overview

This repository contains the reproducible implementation of a manuscript-proposed liver cancer grading framework based on a leakage-safe sequential ensemble architecture.

The proposed framework performs multiclass liver cancer grading using demographic and biomarker information integrated through a sequential ensemble learning strategy:

```text
M1 → M7 → M3
Bagging → CatBoost → ExtraTrees
```

The implementation follows a nested cross-validation strategy with leakage-safe sequential prediction augmentation and SHAP-based explainability analysis.

---

# Proposed Framework

The proposed architecture sequentially augments intermediate predictions:

```text
Original Features
        ↓
M1 (Bagging)
        ↓
Append M1 Predictions
        ↓
M7 (CatBoost)
        ↓
Append M7 Predictions
        ↓
M3 (ExtraTrees)
        ↓
Final Liver Cancer Grade Prediction
```

Unlike conventional stacking methods, this implementation uses sequential feature augmentation with leakage-safe out-of-fold prediction generation.

---

# Key Features

- Leakage-safe nested cross-validation
- Sequential ensemble learning
- Borderline-SMOTE inside training folds only
- Evolutionary hyperparameter optimization
- SHAP explainability analysis
- Multiclass liver cancer grading
- Reproducible manuscript-aligned implementation

---

# Dataset Features

The framework utilizes the following features:

| Feature | Description |
|---|---|
| Age | Patient age |
| Sex | Binary demographic feature |
| Beta_HCG | Beta-human chorionic gonadotropin |
| AFP | Alpha-fetoprotein |
| PD-L1 | Programmed death-ligand 1 |

Target variable:

| Target | Description |
|---|---|
| Grade | Liver cancer grade/class |

---

# Repository Structure

```text
livercancer/
│
├── README.md
├── requirements.txt
│
├── data/
│   └── liver_cancer_data_male_female.csv
│
├── src/
│   ├── 01_preprocessing.py
│   ├── 02_evolutionary_optimization.py
│   ├── 03_sequential_ensemble.py
│   ├── 04_shap_analysis.py
│   └── utils.py
│
├── outputs/
│
├── models/
│
└── results/
```

---

# Methodology

## Data Preprocessing

The preprocessing pipeline performs:

- Removal of redundant columns
- Duplicate removal
- Missing/null sample removal
- Statistical outlier removal using z-score thresholding
- Binary encoding of Sex feature
- Feature standardization

---

## Leakage-Safe Nested Cross-Validation

The framework uses:

- Stratified 5-fold outer cross-validation
- Stratified 5-fold inner cross-validation

To avoid data leakage:

- Borderline-SMOTE is applied only inside training folds
- Out-of-fold predictions are generated using inner cross-validation
- Test folds remain completely isolated during training

---

## Sequential Ensemble Learning

The proposed model uses:

| Model | Classifier |
|---|---|
| M1 | Bagging |
| M7 | CatBoost |
| M3 | ExtraTrees |

Sequential augmentation process:

1. M1 predictions are generated
2. M1 predictions are appended to feature space
3. M7 predictions are generated
4. M7 predictions are appended
5. M3 performs final classification

---

# Evolutionary Hyperparameter Optimization

The repository includes supplementary evolutionary optimization implementation for manuscript reproducibility.

Features:

- Population-based optimization
- Mutation and crossover operations
- 100 evolutionary generations
- Stratified cross-validation evaluation

Optimized models include:

- M1 (Bagging)
- M3 (ExtraTrees)
- M7 (CatBoost)

---

# SHAP Explainability

SHAP analysis is performed using the final ensemble classifier.

Generated explainability outputs include:

- Global SHAP feature importance
- SHAP summary plots
- Grade-wise SHAP analysis
- Biomarker contribution visualization

SHAP visualizations are restricted to original biomarker and demographic features for interpretability consistency.

---

# Installation

## Clone Repository

```bash
git clone https://github.com/rakeshchandrajoshi/livercancer.git

cd livercancer
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Required Python Packages

The repository requires:

```text
numpy
pandas
scipy
scikit-learn
imbalanced-learn
catboost
shap
matplotlib
joblib
```

---

# Dataset Placement

Place the dataset file inside:

```text
data/liver_cancer_data_male_female.csv
```

Required columns:

```text
Age
Sex
Beta_HCG
AFP
PD-L1
Grade
```

---

# Execution Order

Run the following scripts sequentially:

## 1. Data Preprocessing

```bash
python src/01_preprocessing.py
```

Outputs:
- preprocessed_data.pkl
- cleaning_report.json

---

## 2. Evolutionary Hyperparameter Optimization

```bash
python src/02_evolutionary_optimization.py
```

Outputs:
- evolutionary optimization histories
- best parameter files

---

## 3. Sequential Ensemble Evaluation

```bash
python src/03_sequential_ensemble.py
```

Outputs:
- fold-wise metrics
- overall metrics
- confusion matrix
- classification report
- trained final model

---

## 4. SHAP Explainability Analysis

```bash
python src/04_shap_analysis.py
```

Outputs:
- SHAP global feature importance
- SHAP summary plots
- SHAP mean value plots

---

# Generated Outputs

## outputs/

Contains:

- fold-wise evaluation metrics
- classification reports
- confusion matrix
- evolutionary optimization histories

---

## models/

Contains:

- final trained deployment model

---

## results/

Contains:

- SHAP explainability figures
- SHAP feature importance visualizations

---

# Leakage Prevention Strategy

The repository implements several leakage-prevention mechanisms:

- Independent outer and inner cross-validation
- SMOTE applied only within training folds
- Out-of-fold prediction augmentation
- Independent scaling for each outer fold

This ensures scientifically valid performance estimation.

---

# Reproducibility

This repository is designed to provide reproducible support for the associated manuscript methodology, including:

- nested cross-validation,
- sequential ensemble learning,
- evolutionary optimization,
- and explainable AI analysis.

---

# Notes

- The implementation intentionally avoids conventional stacking classifiers.
- Sequential feature augmentation is used instead of meta-learning.
- SHAP plots visualize only original biomarker and demographic features.
- Evolutionary optimization is included as supplementary reproducibility support.

---

# Citation

If you use this repository, please cite the associated manuscript.

---

# License

This repository is provided for academic and research purposes.
