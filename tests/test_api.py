"""Tests for the churn-prediction API endpoints."""

import unittest

from fastapi.testclient import TestClient

from src.api import app
from src.predict import MODEL_PATH, train_and_save_model
from src.preprocessing import load_data


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not MODEL_PATH.exists():
            train_and_save_model()
        cls.client = TestClient(app)
        cls.example_customer = (
            load_data().drop(columns=["customerID", "Churn"]).iloc[0].to_dict()
        )

    def test_health_endpoint(self) -> None:
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_prediction_endpoint_accepts_valid_customer(self) -> None:
        response = self.client.post("/predict", json=self.example_customer)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            set(response.json()),
            {"churn_prediction", "churn_label", "churn_probability"},
        )
