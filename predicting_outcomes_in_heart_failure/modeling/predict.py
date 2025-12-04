from __future__ import annotations

import time

import joblib
from loguru import logger
import numpy as np
import pandas as pd

from predicting_outcomes_in_heart_failure.app.schema import HeartSample
from predicting_outcomes_in_heart_failure.config import (
    FIGURES_DIR,
    INPUT_COLUMNS,
    MODEL_PATH,
    MULTI_CAT,
    NUM_COLS_DEFAULT,
    SCALER_PATH,
)
from predicting_outcomes_in_heart_failure.modeling.explainability import (
    explain_prediction,
    save_shap_waterfall_plot,
)


def preprocessing(sample_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the exact same preprocessing used during training:
    """
    logger.info("Applying preprocessing pipeline for inference...")

    if not (SCALER_PATH.exists() and MODEL_PATH.exists()):
        raise FileNotFoundError("Preprocessing artifacts missing.")

    scaler = joblib.load(SCALER_PATH)
    input_columns = INPUT_COLUMNS
    multi_cat = MULTI_CAT
    num_cols = NUM_COLS_DEFAULT

    logger.debug(f"Loaded scaler from {SCALER_PATH}")
    logger.debug(f"Using {len(input_columns)} input columns")

    if "Sex" in sample_df.columns and "Sex" not in input_columns:
        logger.debug("Dropping column 'Sex' since it's not used by the current model variant.")
        sample_df = sample_df.drop(columns=["Sex"])

    if "Sex" in sample_df.columns and "Sex" in input_columns:
        sample_df["Sex"] = sample_df["Sex"].map({"M": 1, "F": 0}).astype(int)
        logger.debug("Mapped 'Sex' to binary values (M=1, F=0).")

    if "ExerciseAngina" in sample_df.columns and "ExerciseAngina" in input_columns:
        sample_df["ExerciseAngina"] = sample_df["ExerciseAngina"].map({"Y": 1, "N": 0}).astype(int)
        logger.debug("Mapped 'ExerciseAngina' to binary values (Y=1, N=0).")

    present_multi = [c for c in multi_cat if c in sample_df.columns]
    if present_multi:
        logger.debug(f"Performing one-hot encoding on: {present_multi}")
        sample_df = pd.get_dummies(sample_df, columns=present_multi, drop_first=False)

    for col in input_columns:
        if col not in sample_df.columns:
            sample_df[col] = 0
    sample_df = sample_df.reindex(columns=input_columns, fill_value=0)
    logger.debug("Aligned input columns with training feature order.")

    cols_to_scale = [c for c in num_cols if c in sample_df.columns]
    sample_df[cols_to_scale] = scaler.transform(sample_df[cols_to_scale])
    logger.debug(f"Scaled numerical columns: {cols_to_scale}")

    logger.success("Preprocessing completed successfully.")
    return sample_df


def main():
    logger.info("Starting static inference...")

    sample = HeartSample(
        Age=54,
        ChestPainType="ASY",
        RestingBP=140,
        Cholesterol=239,
        FastingBS=0,
        RestingECG="Normal",
        MaxHR=160,
        ExerciseAngina="N",
        Oldpeak=0.0,
        ST_Slope="Up",
    )
    logger.info("Sample created successfully.")

    X_raw = sample.to_dataframe()
    logger.debug(f"Raw input features:\n{X_raw}")
    X = preprocessing(X_raw)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    logger.success(f"Loaded model from {MODEL_PATH}")

    # Perform prediction
    t0 = time.perf_counter()
    y_pred = model.predict(X)[0]
    inference_time = time.perf_counter() - t0
    y_pred = int(y_pred) if np.issubdtype(type(y_pred), np.integer) else y_pred
    result = {
        "prediction": y_pred,
        "inference_time_seconds": inference_time,
    }

    # Explainability
    model = joblib.load(MODEL_PATH)
    model_type = MODEL_PATH.stem
    try:
        logger.info("Computing explanation for the prediction...")
        explanations = explain_prediction(model, X, model_type=model_type, top_k=5)
        result["explanations"] = explanations
        logger.success("Explanation computed successfully.")
    except Exception as e:
        logger.error(f"Failed to compute explanation: {e}")

    try:
        shap_path = FIGURES_DIR / f"shap_waterfall_{model_type}.png"
        saved = save_shap_waterfall_plot(model, X, model_type=model_type, output_path=shap_path)
        if saved is not None:
            result["explanation_plot"] = str(saved)
    except Exception as e:
        logger.error(f"Failed to generate SHAP waterfall plot: {e}")

    logger.info("Inference completed.")
    logger.success(f"Prediction result: {result}")

    return result


if __name__ == "__main__":
    main()
