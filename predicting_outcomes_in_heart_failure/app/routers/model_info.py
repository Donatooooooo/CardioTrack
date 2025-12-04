from http import HTTPStatus
import json
from typing import Any

from fastapi import APIRouter, Request
from loguru import logger
from predicting_outcomes_in_heart_failure.app.utils import construct_response
from predicting_outcomes_in_heart_failure.config import (
    MODEL_PATH,
    REPORTS_DIR,
    TEST_METRICS_DIR,
)

router = APIRouter(tags=["Model"])


@router.get("/model/hyperparameters")
@construct_response
def get_model_hyperparameters(request: Request):
    variant = MODEL_PATH.parent.name
    model_name = MODEL_PATH.stem
    hyperparams_path = REPORTS_DIR / variant / model_name / "cv_parameters.json"
    logger.info(
        f"Looking for hyperparameters file at {hyperparams_path} "
        f"(model={model_name}, variant={variant})"
    )

    if not hyperparams_path.exists():
        logger.warning("Hyperparameters file not found")
        return {
            "message": HTTPStatus.NOT_FOUND.phrase,
            "status-code": HTTPStatus.NOT_FOUND,
            "data": {
                "detail": "Hyperparameters file not found. Run the training pipeline.",
                "model_name": model_name,
                "variant": variant,
                "expected_path": str(hyperparams_path),
            },
        }

    with hyperparams_path.open("r", encoding="utf-8") as f:
        hyperparams_data = json.load(f)

    data: dict[str, Any] = {
        "model_path": str(MODEL_PATH),
        "hyperparameters": hyperparams_data,
    }

    return {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": data,
    }


@router.get("/model/metrics")
@construct_response
def get_model_metrics(request: Request):
    variant = MODEL_PATH.parent.name
    model_name = MODEL_PATH.stem
    metrics_path = TEST_METRICS_DIR / variant / f"{model_name}.json"
    logger.info(
        f"Looking for metrics file at {metrics_path} (model={model_name}, variant={variant})"
    )

    if not metrics_path.exists():
        logger.warning("Metrics file not found")
        return {
            "message": HTTPStatus.NOT_FOUND.phrase,
            "status-code": HTTPStatus.NOT_FOUND,
            "data": {
                "detail": (
                    "Metrics file not found. Run the evaluation pipeline for this model first."
                ),
                "model_name": model_name,
                "variant": variant,
                "expected_path": str(metrics_path),
            },
        }

    with metrics_path.open("r", encoding="utf-8") as f:
        metrics_data = json.load(f)

    data: dict[str, Any] = {
        "model_path": str(MODEL_PATH),
        "model_name": model_name,
        "variant": variant,
        "metrics": metrics_data.get("metrics", metrics_data),
    }

    return {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": data,
    }
