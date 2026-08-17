# Customer Churn Prediction

A compact end-to-end machine-learning project that predicts whether a telecom customer is likely to churn. It demonstrates data inspection, leakage-safe preprocessing, model comparison, model persistence, a FastAPI prediction service, and automated checks.

## Problem

Customer churn is when a customer stops using a service. Identifying customers with a high churn probability can help a telecom company target retention efforts.

The target is `Churn` from the IBM Telco Customer Churn dataset:

- `No` is encoded as `0`
- `Yes` is encoded as `1`
- 1,869 of 7,043 customers (26.54%) are in the churn class

Because churn is the minority class, accuracy alone is not sufficient. This project also reports precision, recall, F1, ROC-AUC, and a confusion matrix, with particular attention to recall for churn.

## Pipeline

```text
Raw CSV
  -> validation / EDA
  -> feature preparation
  -> stratified train/test split
  -> preprocessing + model pipeline
  -> evaluation and model selection
  -> joblib artifact
  -> FastAPI prediction endpoint
```

## Data and preprocessing

The dataset is expected at `data/WA_Fn-UseC_-Telco-Customer-Churn.csv`.

- `customerID` is dropped because it is an identifier, not behavioral information.
- `TotalCharges` is converted with `pd.to_numeric(..., errors="coerce")`; blank values become missing values.
- Numerical values use median imputation followed by `StandardScaler`.
- Categorical values use most-frequent imputation followed by `OneHotEncoder(handle_unknown="ignore")`.
- A `ColumnTransformer` and sklearn `Pipeline` keep preprocessing identical for training and inference. The transformer is fitted only on the training split.

## Models and results

The data is split 80/20 with `stratify=y` and `random_state=42`. The following results are from the held-out test set.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8055 | 0.6572 | 0.5588 | 0.6040 | 0.8419 |
| Random Forest | 0.7757 | 0.6000 | 0.4652 | 0.5241 | 0.8186 |
| HistGradientBoosting | 0.7935 | 0.6326 | 0.5294 | 0.5764 | 0.8355 |

Logistic Regression was selected because it achieved the highest recall, F1, and ROC-AUC in this comparison while remaining easy to explain. Its confusion matrix is `[[926, 109], [165, 209]]`, arranged as `[[TN, FP], [FN, TP]]`.

The selected complete pipeline is saved locally as `models/churn_model.joblib`. Model artifacts are intentionally ignored by Git and can be regenerated at any time.

## Project structure

```text
data/          Dataset CSV
models/        Generated, ignored model artifacts
src/
  eda.py       Basic dataset inspection
  preprocessing.py  Feature preparation and transformer construction
  train.py     Stratified split and Logistic Regression baseline
  evaluate.py  Candidate-model comparison and metrics
  predict.py   Artifact creation, loading, and prediction helper
  api.py       FastAPI service
tests/         Lightweight unittest coverage
```

## Setup

Requires Python 3.12.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Ensure the dataset file is present at:

```text
data/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

## Run the project

```powershell
# Inspect the data
python src/eda.py

# Train and compare candidates on the held-out test set
python src/evaluate.py

# Train the selected pipeline and create models/churn_model.joblib
python src/predict.py

# Run validation checks
python -m unittest discover -s tests -v

# Start the API after the model artifact has been created
uvicorn src.api:app --reload
```

## API

`GET /health` returns:

```json
{"status": "ok"}
```

`POST /predict` accepts the 19 model features. For example:

```json
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 1,
  "PhoneService": "No",
  "MultipleLines": "No phone service",
  "InternetService": "DSL",
  "OnlineSecurity": "No",
  "OnlineBackup": "Yes",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "No",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 29.85,
  "TotalCharges": "29.85"
}
```

Example response from the generated artifact:

```json
{
  "churn_prediction": 1,
  "churn_label": "Yes",
  "churn_probability": 0.6128185504551915
}
```

Interactive API documentation is available at `http://127.0.0.1:8000/docs` while the service is running.

## Engineering decisions

- The split is stratified so the train and test sets preserve the churn ratio.
- Preprocessing is inside the model pipeline to prevent train/inference mismatches and data leakage.
- `handle_unknown="ignore"` lets inference safely handle unseen categorical values.
- The complete pipeline, rather than a separate model and encoder, is persisted with joblib.
- No class weighting was added because the initial unweighted comparison selected Logistic Regression on the relevant held-out metrics.

## Limitations and next steps

- This is a fixed offline dataset, not a deployed or continuously monitored system.
- A default 0.5 classification threshold is used; a real retention program should set its threshold using intervention costs and capacity.
- Further work could include cross-validation, threshold analysis, calibration, feature-importance reporting, monitoring, and periodic retraining.
- The repository currently has no `LICENSE` file; choose and add one before publishing it for reuse.
