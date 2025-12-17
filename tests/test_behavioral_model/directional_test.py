import joblib
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


class TestModelDirectional:
    def test_one_hot_encoding_invariance(self, trained_models, sample_features):
        """
        Check how the model reacts to a change in an one hot encoded feature.
        """

        models = trained_models
        original = sample_features.copy()

        st = [c for c in sample_features.columns if c.startswith("ST_Slope_")]
        active_col = original[st].columns[original[st].iloc[0] == 1][0]
        other_col = [c for c in st if c != active_col][0]

        modified = original.copy()
        modified[active_col] = 0
        modified[other_col] = 1

        for _, model in models.items():
            pred_original = model.predict_proba(original)[0, 1]
            pred_modified = model.predict_proba(modified)[0, 1]
            assert pred_original != pred_modified

    def test_outlier_effect_on_prediction(self, trained_models, sample_features):
        """
        Check how the model reacts to an outlier.
        """

        models = trained_models
        original = sample_features.copy()

        modified = original.copy()
        modified["Oldpeak"] = original["Oldpeak"] + 1.0

        for model_name, model in models.items():
            pred_original = model.predict_proba(original)[0, 1]
            pred_modified = model.predict_proba(modified)[0, 1]

            assert pred_original != pred_modified, f"{model_name}: model not sensitive to outlier"

    def test_max_heart_rate_relationship(self, trained_models, sample_features):
        """
        Lower maximum heart rate achieved should generally increase risk
        """

        models = trained_models

        high_hr = sample_features.copy()
        high_hr["MaxHR"] = 2.0

        low_hr = sample_features.copy()
        low_hr["MaxHR"] = -2.0

        for model_name, model in models.items():
            prob_high = model.predict_proba(high_hr)[0, 1]
            prob_low = model.predict_proba(low_hr)[0, 1]

            assert prob_low >= prob_high - 0.15, (
                f"{model_name}: unexpected directionality for MaxHR. "
            )

    def test_oldpeak_elevation_increases_risk(self, trained_models, sample_features):
        """
        Higher Oldpeak should increase heart disease probability
        """

        models = trained_models

        low_oldpeak = sample_features.copy()
        low_oldpeak["Oldpeak"] = -1.0

        high_oldpeak = sample_features.copy()
        high_oldpeak["Oldpeak"] = 2.0

        for model_name, model in models.items():
            prob_low = model.predict_proba(low_oldpeak)[0, 1]
            prob_high = model.predict_proba(high_oldpeak)[0, 1]

            assert prob_high >= prob_low - 0.15, (
                f"{model_name}: unexpected directionality for Oldpeak. "
            )

    def test_exercise_angina_increases_risk(self, trained_models, sample_features):
        """
        Exercise-induced angina should generally increase heart disease probability
        """

        models = trained_models

        no_angina = sample_features.copy()
        no_angina["ExerciseAngina"] = 0

        with_angina = sample_features.copy()
        with_angina["ExerciseAngina"] = 1

        for model_name, model in models.items():
            if hasattr(model, "predict_proba"):
                prob_no_angina = model.predict_proba(no_angina)[0, 1]
                prob_with_angina = model.predict_proba(with_angina)[0, 1]

                assert prob_with_angina >= prob_no_angina - 0.15, (
                    f"{model_name}: unexpected directionality for ExerciseAngina "
                )
