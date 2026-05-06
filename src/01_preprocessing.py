import os
import json
import joblib

from utils import load_and_preprocess_data


DATA_PATH = "data/liver_cancer_data_male_female.csv"

os.makedirs("outputs", exist_ok=True)

X, y, feature_names_original, sex_encoder, grade_encoder, cleaning_report = (
    load_and_preprocess_data(DATA_PATH)
)

joblib.dump(
    {
        "X": X,
        "y": y,
        "feature_names_original": feature_names_original,
        "sex_encoder": sex_encoder,
        "grade_encoder": grade_encoder,
        "cleaning_report": cleaning_report
    },
    "outputs/preprocessed_data.pkl"
)

with open("outputs/cleaning_report.json", "w") as f:
    json.dump(cleaning_report, f, indent=4, default=str)

print("Preprocessing complete.")
print("Saved: outputs/preprocessed_data.pkl")
print("Saved: outputs/cleaning_report.json")
