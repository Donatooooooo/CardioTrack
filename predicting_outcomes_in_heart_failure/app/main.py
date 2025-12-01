from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import joblib
from loguru import logger

from predicting_outcomes_in_heart_failure.app.routers import (
    general,
    model_info,
    prediction,
    cards
)
from predicting_outcomes_in_heart_failure.config import FIGURES_DIR, MODEL_PATH


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Context manager to handle application lifespan events."""
    if not MODEL_PATH.exists():
        logger.error(f"Model file not found at: {MODEL_PATH}")
        raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")

    logger.info(f"Loading default model from {MODEL_PATH} ...")
    app.state.model = joblib.load(MODEL_PATH)
    logger.success(f"Default model loaded from {MODEL_PATH}")

    try:
        yield
    finally:
        app.state.model = None
        logger.info("Default model cleared on application shutdown")


app = FastAPI(
    title="Heart Failure Prediction",
    description=(
        "This API lets you make predictions on clinical dataset using a RandomForestClassifier."
    ),
    version="0.01",
    lifespan=lifespan,
)

app.mount("/figures", StaticFiles(directory=str(FIGURES_DIR)), name="figures")

# Routers
app.include_router(general.router)
app.include_router(prediction.router)
app.include_router(model_info.router)
app.include_router(cards.router)
