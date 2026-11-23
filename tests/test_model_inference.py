
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"

FEATURES = (
    ["Time"]
    + [f"V{i}" for i in range(1, 29)]
    + ["Amount"]
)

MODEL_FILES = [
    "logistic_regression.joblib",
    "random_forest.joblib",
    "xgboost.joblib",
]


class TestModelInference(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Four sample transactions with all 30 required features.
        cls.sample_data = pd.DataFrame(
            np.zeros((4, len(FEATURES))),
            columns=FEATURES,
        )

    def test_model_files_exist(self):
        for filename in MODEL_FILES:
            with self.subTest(model=filename):
                self.assertTrue(
                    (MODEL_DIR / filename).exists(),
                    f"Missing model file: {filename}",
                )

    def test_models_load_successfully(self):
        for filename in MODEL_FILES:
            with self.subTest(model=filename):
                model = joblib.load(MODEL_DIR / filename)
                self.assertIsNotNone(model)

    def test_models_accept_30_features(self):
        for filename in MODEL_FILES:
            with self.subTest(model=filename):
                model = joblib.load(MODEL_DIR / filename)

                self.assertEqual(model.n_features_in_, 30)

    def test_prediction_probabilities(self):
        for filename in MODEL_FILES:
            with self.subTest(model=filename):
                model = joblib.load(MODEL_DIR / filename)

                probabilities = model.predict_proba(
                    self.sample_data
                )

                self.assertEqual(probabilities.shape, (4, 2))
                self.assertTrue(
                    np.isfinite(probabilities).all()
                )
                self.assertTrue(
                    ((probabilities >= 0) &
                     (probabilities <= 1)).all()
                )

    def test_probabilities_sum_to_one(self):
        for filename in MODEL_FILES:
            with self.subTest(model=filename):
                model = joblib.load(MODEL_DIR / filename)

                probabilities = model.predict_proba(
                    self.sample_data
                )

                np.testing.assert_allclose(
                    probabilities.sum(axis=1),
                    np.ones(4),
                    atol=1e-6,
                )

    def test_threshold_alert_logic(self):
        scores = np.array([0.10, 0.35, 0.50, 0.90])
        threshold = 0.3696

        alerts = scores >= threshold

        expected = np.array([False, False, True, True])

        np.testing.assert_array_equal(alerts, expected)


if __name__ == "__main__":
    unittest.main()
