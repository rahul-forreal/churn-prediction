"""FastAPI service for telecom customer churn predictions."""

from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.pipeline import Pipeline

try:
    from .predict import MODEL_PATH, load_model, predict_customer
except ImportError:
    from predict import MODEL_PATH, load_model, predict_customer


app = FastAPI(title="Customer Churn Prediction API", version="1.0.0")


class CustomerFeatures(BaseModel):
    """Customer attributes expected by the trained churn model."""

    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: str | float


class PredictionResponse(BaseModel):
    """Churn classification and associated positive-class probability."""

    churn_prediction: int
    churn_label: str
    churn_probability: float


@lru_cache
def get_model() -> Pipeline:
    """Load and cache the complete persisted model pipeline."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model artifact not found: {MODEL_PATH}")
    return load_model(MODEL_PATH)


@app.get("/health")
def health() -> dict[str, str]:
    """Report that the API process is available."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerFeatures) -> PredictionResponse:
    """Predict whether one customer is likely to churn."""
    try:
        model = get_model()
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail="Model artifact is unavailable.") from error

    result = predict_customer(customer.model_dump(), model)
    return PredictionResponse(**result)
