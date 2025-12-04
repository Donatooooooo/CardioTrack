from http import HTTPStatus
from typing import Any

from fastapi import APIRouter, Request
from loguru import logger
import pandas as pd
from predicting_outcomes_in_heart_failure.app.schema import HeartSample
from predicting_outcomes_in_heart_failure.app.utils import (
    construct_response,
    get_model_from_state,
)
from predicting_outcomes_in_heart_failure.config import FIGURES_DIR, MODEL_PATH
from predicting_outcomes_in_heart_failure.modeling.explainability import (
    explain_prediction,
    save_shap_waterfall_plot,
)
from predicting_outcomes_in_heart_failure.modeling.predict import preprocessing

router = APIRouter()


@router.post("/predictions", tags=["Prediction"])
@construct_response
def predict(request: Request, payload: HeartSample):
    model = get_model_from_state(request)
    if model is None:
        return {
            "message": HTTPStatus.SERVICE_UNAVAILABLE.phrase,
            "status-code": HTTPStatus.SERVICE_UNAVAILABLE,
            "data": {"detail": "Model is not loaded."},
        }

    X_raw = payload.to_dataframe()
    X = preprocessing(X_raw)
    y_pred = int(model.predict(X)[0])

    data: dict[str, Any] = {
        "input": payload.model_dump(),
        "prediction": y_pred,
    }

    logger.success("Prediction completed successfully for /predictions")
    return {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": data,
    }


@router.post("/batch-predictions", tags=["Prediction"])
@construct_response
def predict_batch(request: Request, payload: list[HeartSample]):
    model = get_model_from_state(request)
    if model is None:
        return {
            "message": HTTPStatus.SERVICE_UNAVAILABLE.phrase,
            "status-code": HTTPStatus.SERVICE_UNAVAILABLE,
            "data": {"detail": "Model is not loaded."},
        }

    X_raw_list = [sample.to_dataframe() for sample in payload]
    X_raw = pd.concat(X_raw_list, ignore_index=True)
    X = preprocessing(X_raw)

    y_pred = [int(y) for y in model.predict(X)]

    results: list[dict[str, Any]] = []
    for idx, (sample, pred) in enumerate(zip(payload, y_pred, strict=True)):
        results.append(
            {
                "index": idx,
                "input": sample.model_dump(),
                "prediction": pred,
            }
        )

    data: dict[str, Any] = {
        "results": results,
        "batch_size": len(results),
    }

    return {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": data,
    }


@router.post("/explanations", tags=["Explainability"])
@construct_response
def explain(request: Request, payload: HeartSample):
    model = get_model_from_state(request)
    if model is None:
        return {
            "message": HTTPStatus.SERVICE_UNAVAILABLE.phrase,
            "status-code": HTTPStatus.SERVICE_UNAVAILABLE,
            "data": {"detail": "Model is not loaded."},
        }

    X_raw = payload.to_dataframe()
    X = preprocessing(X_raw)

    data: dict[str, Any] = {"input": payload.model_dump()}
    model_type = MODEL_PATH.stem

    try:
        logger.info("Computing explanation for default model prediction...")
        explanations = explain_prediction(model, X, model_type=model_type, top_k=5)
        if explanations:
            data["explanations"] = explanations
            logger.success("Explanation computed successfully for default model.")
        else:
            logger.warning("No explanation available for default model.")
    except Exception as e:
        logger.exception(f"Failed to compute explanation: {e}")

    try:
        plot_path = FIGURES_DIR / f"shap_waterfall_default_{model_type}.png"
        saved_path = save_shap_waterfall_plot(
            model=model,
            X=X,
            model_type=model_type,
            output_path=plot_path,
        )
        if saved_path is not None:
            data["explanation_plot_url"] = f"/figures/{saved_path.name}"
    except Exception as e:
        logger.exception(f"Failed to generate explanation plot: {e}")

    logger.success("Explanation completed successfully for /explanations")
    return {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": data,
    }
