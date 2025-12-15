import joblib
import numpy as np
import pandas as pd
from predicting_outcomes_in_heart_failure.config import MODELS_DIR, NOSEX_CSV, TARGET_COL
import pytest


@pytest.fixture
def sample_features():
    df = pd.read_csv(NOSEX_CSV).copy()
    return df.iloc[[0]].drop(columns=[TARGET_COL])


@pytest.fixture
def trained_models():
    """
    Load all saved models.
    """
    models = {}
    models_path = MODELS_DIR / "nosex"

    for model_file in models_path.iterdir():
        if model_file.suffix == ".joblib":
            model_name = model_file.stem
            models[model_name] = joblib.load(model_file)

    return models


class TestModelInvariance:
    def test_prediction_determinism(self, trained_models, sample_features):
        """
        Same input should always produce same prediction.
        """

        models = trained_models
        for model_name, model in models.items():
            pred1 = model.predict(sample_features)
            pred2 = model.predict(sample_features)
            pred3 = model.predict(sample_features)

            assert np.array_equal(pred1, pred2), f"{model_name}: non-deterministic predictions"
            assert np.array_equal(pred2, pred3), f"{model_name}: non-deterministic predictions"

    def test_uniform_scaling_invariance(self, trained_models, sample_features):
        """
        Uniform scaling should always produce same prediction.
        """

        models = trained_models
        original = sample_features.copy()

        modified = original.copy() * 1.1

        for model_name, model in models.items():
            pred_original = model.predict(original)
            pred_modified = model.predict(modified)
            assert np.array_equal(pred_original, pred_modified), (
                f"{model_name}: prediction changed after uniform scaling"
            )
