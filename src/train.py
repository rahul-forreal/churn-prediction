"""Train a leakage-safe Logistic Regression baseline for churn prediction."""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

try:
    from .preprocessing import (
        build_preprocessor,
        load_data,
        prepare_features_and_target,
    )
except ImportError:
    from preprocessing import build_preprocessor, load_data, prepare_features_and_target


TEST_SIZE = 0.2
RANDOM_STATE = 42


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create a reproducible split that preserves the churn-class ratio."""
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def build_baseline_pipeline(X_train: pd.DataFrame) -> Pipeline:
    """Create an unfitted preprocessing and Logistic Regression pipeline."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X_train)),
            ("model", LogisticRegression(max_iter=1_000, random_state=RANDOM_STATE)),
        ]
    )


def train_baseline_model(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """Fit the baseline pipeline using training data only."""
    baseline_pipeline = build_baseline_pipeline(X_train)
    baseline_pipeline.fit(X_train, y_train)
    return baseline_pipeline


if __name__ == "__main__":
    raw_data = load_data()
    features, target = prepare_features_and_target(raw_data)
    X_train, X_test, y_train, y_test = split_data(features, target)
    baseline_pipeline = train_baseline_model(X_train, y_train)

    print(f"Training features: {X_train.shape}")
    print(f"Test features: {X_test.shape}")
    print("Training target distribution:")
    print(y_train.value_counts(normalize=True).sort_index())
    print("Test target distribution:")
    print(y_test.value_counts(normalize=True).sort_index())
    print(f"Baseline pipeline trained: {baseline_pipeline.named_steps['model']}")
