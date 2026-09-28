from pathlib import Path
import sys

import pandas as pd
import shap
from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / "src"))


from models.process_xgboost import (
    FEATURE_COLUMNS,
    MODEL_PATH,
)


def load_model():
    model = XGBClassifier()
    model.load_model(MODEL_PATH)
    return model


def explain_process_prediction(process_values, top_n=5):
    """
    Explain a process-quality prediction using SHAP.

    Returns the top contributing process parameters
    for the predicted quality class.
    """

    model = load_model()

    X = pd.DataFrame(
        [process_values],
        columns=FEATURE_COLUMNS
    )

    # Create SHAP explainer for the trained XGBoost model
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)

    predicted_class = int(model.predict(X)[0])

    # For multiclass XGBoost, select SHAP values
    # belonging to the predicted class.
    class_shap_values = shap_values.values[0, :, predicted_class]

    explanation_df = pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "shap_value": class_shap_values,
        "absolute_shap": abs(class_shap_values),
        "value": process_values,
    })

    explanation_df = explanation_df.sort_values(
        "absolute_shap",
        ascending=False
    )

    top_features = explanation_df.head(top_n)

    explanations = []

    for _, row in top_features.iterrows():
        explanations.append({
            "feature": row["feature"],
            "value": float(row["value"]),
            "shap_value": float(row["shap_value"]),
        })

    return explanations


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
        18.75,
    ]

    explanations = explain_process_prediction(
        sample_values,
        top_n=5
    )

    print("=== PROCESS EXPLANATION ===")
    print()

    for item in explanations:
        direction = (
            "supports prediction"
            if item["shap_value"] > 0
            else "pushes against prediction"
        )

        print(
            f"{item['feature']}: "
            f"{item['value']:.4f} | "
            f"SHAP = {item['shap_value']:.4f} | "
            f"{direction}"
        )