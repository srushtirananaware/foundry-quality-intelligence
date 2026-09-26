from pathlib import Path

import pandas as pd
from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "process_quality_xgb.json"


FEATURE_COLUMNS = [
    "Melt temperature",
    "Mold temperature",
    "time_to_fill",
    "ZDx - Plasticizing time",
    "ZUx - Cycle time",
    "SKx - Closing force",
    "SKs - Clamping force peak value",
    "Ms - Torque peak value current cycle",
    "Mm - Torque mean value current cycle",
    "APSs - Specific back pressure peak value",
    "APVs - Specific injection pressure peak value",
    "CPn - Screw position at the end of hold pressure",
    "SVo - Shot volume",
]


def load_model():
    model = XGBClassifier()
    model.load_model(MODEL_PATH)
    return model


def predict_quality(process_values):
    model = load_model()

    X = pd.DataFrame(
        [process_values],
        columns=FEATURE_COLUMNS
    )

    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]

    predicted_quality = int(prediction) + 1

    quality_probabilities = {
        f"Quality {i + 1}": float(probabilities[i])
        for i in range(4)
    }

    return predicted_quality, quality_probabilities


if __name__ == "__main__":

    sample_values = [
        106.0,
        81.2,
        7.0,
        3.2,
        75.0,
        900.0,
        920.0,
        117.0,
        105.0,
        146.2,
        910.0,
        8.8,
        18.75
    ]

    predicted_quality, probabilities = predict_quality(
        sample_values
    )

    print("Predicted quality:", predicted_quality)

    print("\nQuality probabilities:")

    for quality, probability in probabilities.items():
        print(f"{quality}: {probability:.2%}")