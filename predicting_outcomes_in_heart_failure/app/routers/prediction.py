from http import HTTPStatus
import time
from typing import Any

from fastapi import APIRouter, Request
from loguru import logger
import pandas as pd
from predicting_outcomes_in_heart_failure.app.deepchecks_monitoring import (
    production_data_collector as pdc,
)
from predicting_outcomes_in_heart_failure.app.monitoring import (
    batch_size_histogram,
    explanation_counter,
    model_error_counter,
    prediction_counter,
    prediction_processing_time,
    prediction_result_counter,
)
from predicting_outcomes_in_heart_failure.app.schema import HeartSample
from predicting_outcomes_in_heart_failure.app.utils import (
    construct_response,
    get_model_from_state,
)
from predicting_outcomes_in_heart_failure.config import (
    FIGURES_DIR,
    INPUT_COLUMNS,
    MODEL_PATH,
    PRODUCTION_CSV_PATH,
)
from predicting_outcomes_in_heart_failure.modeling.explainability import (
    explain_prediction,
    save_shap_waterfall_plot,
)
from predicting_outcomes_in_heart_failure.modeling.predict import preprocessing

router = APIRouter()


@router.post("/predictions", tags=["Prediction"])
@construct_response
def predict(request: Request, payload: HeartSample):
    prediction_counter.labels(prediction_type="single", endpoint="/predictions").inc()

    model = get_model_from_state(request)
    if model is None:
        model_error_counter.labels(error_type="model_not_loaded", endpoint="/predictions").inc()
        return {
            "message": HTTPStatus.SERVICE_UNAVAILABLE.phrase,
            "status-code": HTTPStatus.SERVICE_UNAVAILABLE,
            "data": {"detail": "Model is not loaded."},
        }

    start_time = time.time()
    try:
        X_raw = payload.to_dataframe()
        X = preprocessing(X_raw)
        y_pred = int(model.predict(X)[0])

        pdc.append_predictions_to_csv(
            csv_path=PRODUCTION_CSV_PATH,
            endpoint="/predictions",
            X=X,
            y_pred=y_pred,
            feature_columns=list(INPUT_COLUMNS),
        )

        processing_time = time.time() - start_time
        prediction_processing_time.labels(
            prediction_type="single", endpoint="/predictions"
        ).observe(processing_time)

        prediction_result_counter.labels(
            prediction_class=str(y_pred), endpoint="/predictions"
        ).inc()

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
    except Exception as e:
        model_error_counter.labels(error_type="prediction_error", endpoint="/predictions").inc()
        logger.exception(f"Prediction error: {e}")
        raise


@router.post("/batch-predictions", tags=["Prediction"])
@construct_response
def predict_batch(request: Request, payload: list[HeartSample]):
    prediction_counter.labels(prediction_type="batch", endpoint="/batch-predictions").inc()
    batch_size = len(payload)
    batch_size_histogram.labels(endpoint="/batch-predictions").observe(batch_size)

    model = get_model_from_state(request)
    if model is None:
        model_error_counter.labels(
            error_type="model_not_loaded", endpoint="/batch-predictions"
        ).inc()
        return {
            "message": HTTPStatus.SERVICE_UNAVAILABLE.phrase,
            "status-code": HTTPStatus.SERVICE_UNAVAILABLE,
            "data": {"detail": "Model is not loaded."},
        }

    start_time = time.time()
    try:
        X_raw_list = [sample.to_dataframe() for sample in payload]
        X_raw = pd.concat(X_raw_list, ignore_index=True)
        X = preprocessing(X_raw)

        y_pred = [int(y) for y in model.predict(X)]

        pdc.append_predictions_to_csv(
            csv_path=PRODUCTION_CSV_PATH,
            endpoint="/batch-predictions",
            X=X,
            y_pred=y_pred,
            feature_columns=list(INPUT_COLUMNS),
        )

        processing_time = time.time() - start_time
        prediction_processing_time.labels(
            prediction_type="batch", endpoint="/batch-predictions"
        ).observe(processing_time)

        for pred in y_pred:
            prediction_result_counter.labels(
                prediction_class=str(pred), endpoint="/batch-predictions"
            ).inc()

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
    except Exception as e:
        model_error_counter.labels(
            error_type="prediction_error", endpoint="/batch-predictions"
        ).inc()
        logger.exception(f"Batch prediction error: {e}")
        raise


@router.post("/explanations", tags=["Explainability"])
@construct_response
def explain(request: Request, payload: HeartSample):
    model = get_model_from_state(request)
    if model is None:
        explanation_counter.labels(status="error_model_not_loaded", endpoint="/explanations").inc()
        return {
            "message": HTTPStatus.SERVICE_UNAVAILABLE.phrase,
            "status-code": HTTPStatus.SERVICE_UNAVAILABLE,
            "data": {"detail": "Model is not loaded."},
        }

    X_raw = payload.to_dataframe()
    X = preprocessing(X_raw)

    data: dict[str, Any] = {"input": payload.model_dump()}
    model_type = MODEL_PATH.stem
    explanation_success = False

    try:
        logger.info("Computing explanation for default model prediction...")
        explanations = explain_prediction(model, X, model_type=model_type, top_k=5)
        if explanations:
            data["explanations"] = explanations
            logger.success("Explanation computed successfully for default model.")
            explanation_success = True
        else:
            logger.warning("No explanation available for default model.")
    except Exception as e:
        logger.exception(f"Failed to compute explanation: {e}")
        explanation_counter.labels(
            status="error_computation_failed", endpoint="/explanations"
        ).inc()

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
        explanation_counter.labels(
            status="error_plot_generation_failed", endpoint="/explanations"
        ).inc()

    if explanation_success:
        explanation_counter.labels(status="success", endpoint="/explanations").inc()

    logger.success("Explanation completed successfully for /explanations")
    return {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": data,
    }
