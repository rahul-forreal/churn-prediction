"""Reusable preprocessing utilities for the telecom churn dataset."""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
TARGET_COLUMN = "Churn"
ID_COLUMN = "customerID"
TARGET_MAPPING = {"No": 0, "Yes": 1}


def load_data(data_path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Load the raw Telco churn dataset from disk."""
    return pd.read_csv(data_path)


def prepare_features_and_target(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Clean raw data and separate predictor features from the churn target.

    Blank values in ``TotalCharges`` become missing values and are imputed by
    the numerical pipeline after the train/test split. ``customerID`` is an
    identifier rather than a predictive feature, so it is excluded.
    """
    required_columns = {ID_COLUMN, "TotalCharges", TARGET_COLUMN}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing_columns)}"
        )

    cleaned_df = df.copy()
    cleaned_df["TotalCharges"] = pd.to_numeric(
        cleaned_df["TotalCharges"], errors="coerce"
    )

    X = cleaned_df.drop(columns=[ID_COLUMN, TARGET_COLUMN])
    y = cleaned_df[TARGET_COLUMN].map(TARGET_MAPPING)

    if y.isna().any():
        unexpected_labels = cleaned_df.loc[y.isna(), TARGET_COLUMN].unique()
        raise ValueError(f"Unexpected churn labels: {unexpected_labels.tolist()}")

    return X, y.astype("int64")


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Create an unfitted transformer for numerical and categorical features."""
    numeric_features = X.select_dtypes(include="number").columns.tolist()
    categorical_features = X.select_dtypes(exclude="number").columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )


if __name__ == "__main__":
    raw_data = load_data()
    features, target = prepare_features_and_target(raw_data)

    print(f"Feature shape: {features.shape}")
    print(f"Target shape: {target.shape}")
    print(f"Missing TotalCharges values: {features['TotalCharges'].isna().sum()}")
    print("Target distribution:")
    print(target.value_counts().sort_index())
    print("Preprocessor created successfully. Fit it only on training data.")
