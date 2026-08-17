"""Persist and use the selected churn-prediction pipeline."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

try:
    from .preprocessing import load_data, prepare_features, prepare_features_and_target
    from .train import split_data, train_baseline_model
except ImportError:
    from preprocessing import load_data, prepare_features, prepare_features_and_target
    from train import split_data, train_baseline_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "churn_model.joblib"


def train_and_save_model(model_path: str | Path = MODEL_PATH) -> Path:
    """Train the selected Logistic Regression pipeline and save it with joblib."""
    raw_data = load_data()
    features, target = prepare_features_and_target(raw_data)
    X_train, _, y_train, _ = split_data(features, target)

    model = train_baseline_model(X_train, y_train)
    artifact_path = Path(model_path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, artifact_path)
    return artifact_path


def load_model(model_path: str | Path = MODEL_PATH) -> Pipeline:
    """Load a persisted preprocessing-and-model pipeline."""
    return joblib.load(model_path)


def predict_customer(
    customer: dict[str, object], model: Pipeline
) -> dict[str, int | str | float]:
    """Return a churn class, readable label, and positive-class probability."""
    features = prepare_features(pd.DataFrame([customer]))
    prediction = int(model.predict(features)[0])
    probability = float(model.predict_proba(features)[0, 1])

    return {
        "churn_prediction": prediction,
        "churn_label": "Yes" if prediction == 1 else "No",
        "churn_probability": probability,
    }


if __name__ == "__main__":
    saved_model_path = train_and_save_model()
    churn_model = load_model(saved_model_path)

    example_customer = load_data().drop(columns="Churn").iloc[0].to_dict()
    prediction_result = predict_customer(example_customer, churn_model)

    print(f"Saved model: {saved_model_path}")
    print(prediction_result)
