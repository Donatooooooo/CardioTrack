from http import HTTPStatus

from fastapi import APIRouter, Request
from loguru import logger
from predicting_outcomes_in_heart_failure.app.utils import construct_response

router = APIRouter(tags=["General"])


@router.get("/")
@construct_response
def index(request: Request):
    """Root endpoint."""
    logger.info("General requested")
    return {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {"message": "Welcome to Heart Failure Predictor!"},
    }
