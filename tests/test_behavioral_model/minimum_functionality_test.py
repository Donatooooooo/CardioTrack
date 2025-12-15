import joblib
import numpy as np
import pandas as pd
from predicting_outcomes_in_heart_failure.config import (
    MODELS_DIR,
    NOSEX_CSV,
    TARGET_COL,
)
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


class TestModelMinimumFunctionality:
    def test_predict_returns_binary_array(self, trained_models, sample_features):
        """
        Models should return binary predictions (0 or 1) for heart disease.
        """

        models = trained_models
        for model_name, model in models.items():
            predictions = model.predict(sample_features)
            assert predictions.shape == (1,), f"{model_name}: wrong shape"
            assert predictions[0] in [0, 1], f"{model_name}: prediction not binary"

    def test_predict_proba_returns_valid_probabilities(self, trained_models, sample_features):
        """
        Models with predict_proba should return valid probability distributions.
        """

        models = trained_models
        for model_name, model in models.items():
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(sample_features)
                assert proba.shape == (1, 2), f"{model_name}: wrong probability shape"
                assert np.allclose(proba.sum(axis=1), 1.0), (
                    f"{model_name}: probabilities don't sum to 1"
                )
                assert np.all(proba >= 0) and np.all(proba <= 1), (
                    f"{model_name}: invalid probability values"
                )

    def test_batch_prediction_consistency(self, trained_models, sample_features):
        """
        Predictions on batches should be consistent with individual predictions.
        """

        models = trained_models

        batch = pd.concat([sample_features] * 5, ignore_index=True)
        for model_name, model in models.items():
            batch_pred = model.predict(batch)
            single_pred = model.predict(sample_features)

            assert len(batch_pred) == 5, f"{model_name}: wrong batch size"
            assert np.all(batch_pred == single_pred[0]), (
                f"{model_name}: inconsistent batch predictions"
            )

    def test_correct_number_of_features(self, trained_models, sample_features):
        """
        Models should expect exactly 17 features after preprocessing
        """

        models = trained_models
        expected_n_features = 17

        assert len(sample_features.columns) == expected_n_features, (
            f"Sample has {len(sample_features.columns)} columns, expected {expected_n_features}"
        )

        for _, model in models.items():
            _ = model.predict(sample_features)

            with pytest.raises((ValueError, IndexError)):
                wrong_features = sample_features.iloc[:, :10]
                model.predict(wrong_features)

    def test_invalid_one_hot_encoding_detection(self, trained_models, sample_features):
        """
        Test behavior when one-hot encoding is invalid
        """

        models = trained_models

        invalid_none = sample_features.copy()
        invalid_none["ChestPainType_ASY"] = 0
        invalid_none["ChestPainType_ATA"] = 0
        invalid_none["ChestPainType_NAP"] = 0
        invalid_none["ChestPainType_TA"] = 0

        for model_name, model in models.items():
            pred = model.predict(invalid_none)
            assert pred[0] in [0, 1], f"{model_name}: should still return valid binary prediction"
