from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import joblib
import gradio as gr
from loguru import logger
import httpx
import pandas as pd

from predicting_outcomes_in_heart_failure.app.routers import cards, general, model_info, prediction
from predicting_outcomes_in_heart_failure.app.wrapper import load_dataset_card, load_hyperparameters, load_metrics, load_model_card, predict_single, predict_batch
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
    title="CardioTrack's Model Space - Heart Failure Prediction",
    version="0.01",
    lifespan=lifespan,
)

app.mount("/figures", StaticFiles(directory=str(FIGURES_DIR)), name="figures")

# Routers
app.include_router(general.router)
app.include_router(prediction.router)
app.include_router(model_info.router)
app.include_router(cards.router)


# UI
with gr.Blocks(title="CardioTrack") as io:
    gr.Markdown(
        """
        # 🫀 CardioTrack's Model Space - Heart Failure Prediction
        Choose an area to access the platform's features.
        """
    )
    
    with gr.Tabs():
        with gr.TabItem("Bonjour"):
            gr.Markdown("Coming soon...")
            
        with gr.TabItem("Single Prediction"):
            gr.Markdown("### Enter patient data for prediction")
            
            with gr.Row():
                with gr.Column():
                    age = gr.Slider(minimum=20, maximum=100, step=1, label="Age", value=60)
                    resting_bp = gr.Slider(minimum=80, maximum=200, step=1, label="Resting Blood Pressure (mm Hg)", value=120)
                    cholesterol = gr.Slider(minimum=0, maximum=600, step=1, label="Cholesterol (mg/dL)", value=200)
                    max_hr = gr.Slider(minimum=60, maximum=220, step=1, label="Max Heart Rate", value=150)
                    oldpeak = gr.Slider(minimum=-3.0, maximum=7.0, step=0.1, label="Oldpeak (ST Depression)", value=1.0)
                
                with gr.Column():
                    chest_pain_type = gr.Dropdown(
                        choices=["TA", "ATA", "NAP", "ASY"],
                        label="Chest Pain Type",
                        value="ASY"
                    )
                    fasting_bs = gr.Dropdown(
                        choices=[0, 1],
                        label="Fasting Blood Sugar (0: <=120 mg/dL, 1: >120 mg/dL)",
                        value=0
                    )
                    resting_ecg = gr.Dropdown(
                        choices=["Normal", "ST", "LVH"],
                        label="Resting ECG",
                        value="Normal"
                    )
                    exercise_angina = gr.Dropdown(
                        choices=["Y", "N"],
                        label="Exercise Angina",
                        value="N"
                    )
                    st_slope = gr.Dropdown(
                        choices=["Up", "Flat", "Down"],
                        label="ST Slope",
                        value="Flat"
                    )
            
            predict_btn = gr.Button("Predict", variant="primary")
            single_output = gr.Markdown(label="Prediction Result")
            
            predict_btn.click(
                fn=predict_single,
                inputs=[age, chest_pain_type, resting_bp, cholesterol,
                    fasting_bs, resting_ecg, max_hr, exercise_angina,
                    oldpeak, st_slope],
                outputs=single_output
            )
        
        with gr.TabItem("Batch Prediction"):
            gr.Markdown("### Upload a CSV file for batch predictions")
            gr.Markdown("The CSV should contain columns: Age, ChestPainType, RestingBP, Cholesterol, FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope")
            
            file_input = gr.File(label="Upload CSV", file_types=[".csv"])
            batch_predict_btn = gr.Button("Predict Batch", variant="primary")
            batch_output = gr.Dataframe(label="Batch Prediction Results")
            
            batch_predict_btn.click(
                fn=predict_batch,
                inputs=file_input,
                outputs=batch_output
            )

        with gr.TabItem("ModelCard"):
            model_card_content = gr.Markdown("Loading...")
            
            io.load(
                fn=load_model_card,
                inputs=None,
                outputs=model_card_content
            )

        with gr.TabItem("DatasetCard"):
            dataset_card_content = gr.Markdown("Loading...")
            
            io.load(
                fn=load_dataset_card,
                inputs=None,
                outputs=dataset_card_content
            )
        
        with gr.TabItem("Hyperparameters"):
            gr.Markdown("## Model Hyperparameters")
            hyperparams_content = gr.Markdown("Loading...")
            
            io.load(
                fn=load_hyperparameters,
                inputs=None,
                outputs=hyperparams_content
            )
                        
        with gr.TabItem("Evaluation Metrics"):
            gr.Markdown("## Model Performance Metrics")
            metrics_content = gr.Markdown("Loading...")
            
            io.load(
                fn=load_metrics,
                inputs=None,
                outputs=metrics_content
            )

app = gr.mount_gradio_app(app, io, path="/ui")