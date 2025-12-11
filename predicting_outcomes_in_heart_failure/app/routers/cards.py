from http import HTTPStatus

from fastapi import APIRouter, HTTPException, Request
from loguru import logger
from predicting_outcomes_in_heart_failure.app.utils import construct_response
from predicting_outcomes_in_heart_failure.config import CARD_PATHS

router = APIRouter(tags=["Cards"])


@router.get("/cards/{card_type}")
@construct_response
def card(request: Request, card_type: str):
    """Return card information.
    card_type = dataset_card / model_card
    """
    logger.info(f"Received /cards/{card_type} request")

    # Normalizza il card_type per gestire eventuali varianti
    card_type = card_type.lower().replace("-", "_")

    path = CARD_PATHS.get(card_type)
    if path is None:
        logger.warning(f"Unsupported card_type requested: {card_type}")
        raise HTTPException(
            status_code=HTTPStatus.NOT_FOUND,
            detail=f"Card type '{card_type}' not supported."
            + f" Valid types: {', '.join(CARD_PATHS.keys())}",
        )

    try:
        with open(path, encoding="utf-8") as f:
            card_content = f.read()

        logger.success(f"{path} loaded successfully")

        return {
            "message": HTTPStatus.OK.phrase,
            "status-code": HTTPStatus.OK.value,
            "data": {
                "card_type": card_type,
                "path": str(path),
                "card_lines": card_content.split("\n"),
            },
        }

    except Exception as e:
        logger.exception(f"Failed to load card content from {path}: {e}")
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail=f"Error reading card file: {e}",
        ) from e
