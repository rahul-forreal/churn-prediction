"""Tests for dataset preparation and feature transformation."""

import unittest

import numpy as np

from src.preprocessing import build_preprocessor, load_data, prepare_features_and_target


class PreprocessingTests(unittest.TestCase):
    def test_preprocessing_transforms_clean_feature_matrix(self) -> None:
        raw_data = load_data()
        features, target = prepare_features_and_target(raw_data)

        transformed_features = build_preprocessor(features).fit_transform(
            features.head(100)
        )

        self.assertNotIn("customerID", features.columns)
        self.assertEqual(set(target.unique()), {0, 1})
        self.assertEqual(transformed_features.shape[0], 100)
        self.assertTrue(np.isfinite(transformed_features).all())
