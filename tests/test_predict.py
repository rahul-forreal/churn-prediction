"""Tests for the persisted model and prediction helper."""

import unittest

from src.predict import MODEL_PATH, load_model, predict_customer, train_and_save_model
from src.preprocessing import load_data


class PredictionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not MODEL_PATH.exists():
            train_and_save_model()

    def test_model_artifact_can_be_loaded(self) -> None:
        model = load_model()

        self.assertTrue(MODEL_PATH.exists())
        self.assertTrue(hasattr(model, "predict"))
        self.assertTrue(hasattr(model, "predict_proba"))

    def test_prediction_returns_expected_fields(self) -> None:
        model = load_model()
        example_customer = load_data().drop(columns="Churn").iloc[0].to_dict()

        result = predict_customer(example_customer, model)

        self.assertEqual(
            set(result),
            {"churn_prediction", "churn_label", "churn_probability"},
        )
        self.assertIn(result["churn_prediction"], {0, 1})
        self.assertIn(result["churn_label"], {"No", "Yes"})
        self.assertGreaterEqual(result["churn_probability"], 0.0)
        self.assertLessEqual(result["churn_probability"], 1.0)
