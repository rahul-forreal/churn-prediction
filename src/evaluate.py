"""Compare churn-model candidates using a single held-out test set."""

from collections.abc import Iterable

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

try:
    from .preprocessing import build_preprocessor, load_data, prepare_features_and_target
    from .train import RANDOM_STATE, split_data
except ImportError:
    from preprocessing import build_preprocessor, load_data, prepare_features_and_target
    from train import RANDOM_STATE, split_data


def build_model_pipelines(X_train: pd.DataFrame) -> dict[str, Pipeline]:
    """Create unfitted pipelines for the agreed churn-model candidates."""
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1_000, random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_STATE
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            random_state=RANDOM_STATE
        ),
    }

    return {
        name: Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(X_train)),
                ("model", model),
            ]
        )
        for name, model in models.items()
    }


def evaluate_model(
    model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series
) -> dict[str, float | list[list[int]]]:
    """Calculate classification metrics with churn as the positive class."""
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
    }


def compare_models(
    pipelines: Iterable[tuple[str, Pipeline]],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, dict[str, float | list[list[int]]]]:
    """Fit each candidate on training data and evaluate it once on test data."""
    results = {}
    for name, pipeline in pipelines:
        pipeline.fit(X_train, y_train)
        results[name] = evaluate_model(pipeline, X_test, y_test)
    return results


def print_results(results: dict[str, dict[str, float | list[list[int]]]]) -> None:
    """Print a concise, readable comparison of evaluation results."""
    for name, metrics in results.items():
        print(f"\nModel: {name}")
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1:        {metrics['f1']:.4f}")
        print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
        print("Confusion Matrix [[TN, FP], [FN, TP]]:")
        print(metrics["confusion_matrix"])


if __name__ == "__main__":
    raw_data = load_data()
    features, target = prepare_features_and_target(raw_data)
    X_train, X_test, y_train, y_test = split_data(features, target)

    candidate_pipelines = build_model_pipelines(X_train)
    evaluation_results = compare_models(
        candidate_pipelines.items(), X_train, y_train, X_test, y_test
    )
    print_results(evaluation_results)
