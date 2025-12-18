from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import gradio as gr
import joblib
from loguru import logger

from predicting_outcomes_in_heart_failure.app.routers import cards, model_info, prediction
from predicting_outcomes_in_heart_failure.app.utils import load_page, update_patient_index_choices
from predicting_outcomes_in_heart_failure.app.wrapper import Wrapper
from predicting_outcomes_in_heart_failure.config import FIGURES_DIR, MODEL_PATH
from predicting_outcomes_in_heart_failure.app.monitoring import instrumentator

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


if not FIGURES_DIR.exists():
    logger.warning(f"Figures directory {FIGURES_DIR} does not exist. Creating it.")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/figures", StaticFiles(directory=str(FIGURES_DIR)), name="figures")
app.include_router(prediction.router)
app.include_router(model_info.router)
app.include_router(cards.router)

instrumentator.instrument(app).expose(app, include_in_schema=False, should_gzip=True)

# UI Definition
with gr.Blocks(title="CardioTrack") as io:
    gr.Markdown(
        """
        # 🫀 CardioTrack's Model Space - Heart Failure Diagnosis
        Choose an area to access the platform's features.
        """
    )

    with gr.Tabs():
        with gr.TabItem("Single Diagnosis"):
            gr.Markdown("### Enter patient data for the diagnosis")

            with gr.Row():
                with gr.Column():
                    age = gr.Slider(minimum=20, maximum=100, step=1, label="Age", value=60)
                    resting_bp = gr.Slider(
                        minimum=80,
                        maximum=200,
                        step=1,
                        label="Resting Blood Pressure (mm Hg)",
                        value=120,
                    )
                    cholesterol = gr.Slider(
                        minimum=0, maximum=600, step=1, label="Cholesterol (mg/dL)", value=200
                    )
                    max_hr = gr.Slider(
                        minimum=60, maximum=220, step=1, label="Max Heart Rate", value=150
                    )
                    oldpeak = gr.Slider(
                        minimum=-3.0,
                        maximum=7.0,
                        step=0.1,
                        label="Oldpeak (ST Depression)",
                        value=1.0,
                    )

                with gr.Column():
                    chest_pain_type = gr.Dropdown(
                        choices=["TA", "ATA", "NAP", "ASY"], label="Chest Pain Type", value="ASY"
                    )
                    fasting_bs = gr.Dropdown(
                        choices=[0, 1],
                        label="Fasting Blood Sugar (0: <=120 mg/dL, 1: >120 mg/dL)",
                        value=0,
                    )
                    resting_ecg = gr.Dropdown(
                        choices=["Normal", "ST", "LVH"], label="Resting ECG", value="Normal"
                    )
                    exercise_angina = gr.Dropdown(
                        choices=["Y", "N"], label="Exercise Angina", value="N"
                    )
                    st_slope = gr.Dropdown(
                        choices=["Up", "Flat", "Down"], label="ST Slope", value="Flat"
                    )

            predict_btn = gr.Button("Analyze", variant="primary")
            single_output = gr.Markdown(label="Result")

            explanation_img = gr.Image(label="Explanation", type="filepath", visible=True)

            predict_btn.click(
                fn=Wrapper.prediction_with_explanation,
                inputs=[
                    age,
                    chest_pain_type,
                    resting_bp,
                    cholesterol,
                    fasting_bs,
                    resting_ecg,
                    max_hr,
                    exercise_angina,
                    oldpeak,
                    st_slope,
                ],
                outputs=[single_output, explanation_img],
            )

        with gr.TabItem("Group Diagnosis"):
            gr.Markdown("### Upload a CSV file for analize multiple subjects")
            gr.Markdown(
                "The CSV should contain columns: Age, ChestPainType, RestingBP, Cholesterol,"
                + "FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope"
            )

            file_input = gr.File(label="Upload CSV", file_types=[".csv"])
            batch_predict_btn = gr.Button("Analyze Group", variant="primary")
            batch_output = gr.Dataframe(label="Results")

            batch_predict_btn.click(
                fn=Wrapper.batch_prediction, inputs=file_input, outputs=batch_output
            )

            gr.Markdown("### Explain a specific patient from the group")

            patient_index = gr.Dropdown(
                label="Patient index (0-based)",
                choices=[],
                interactive=True,
            )

            batch_output.change(
                fn=update_patient_index_choices,
                inputs=batch_output,
                outputs=patient_index,
            )

            batch_explain_btn = gr.Button("Explain selected patient", variant="secondary")

            batch_explanation_img = gr.Image(
                label="Explanation",
                type="filepath",
            )

            batch_explain_btn.click(
                fn=Wrapper.batch_explanation,
                inputs=[file_input, patient_index],
                outputs=batch_explanation_img,
            )

        with gr.TabItem("ModelCard"):
            io = load_page(io, Wrapper.get_model_card)

        with gr.TabItem("DatasetCard"):
            io = load_page(io, Wrapper.get_dataset_card)

        with gr.TabItem("Hyperparameters"):
            gr.Markdown("## Model Hyperparameters")
            io = load_page(io, Wrapper.get_hyperparameters)

        with gr.TabItem("Evaluation Metrics"):
            gr.Markdown("## Model Performance Metrics")
            io = load_page(io, Wrapper.get_metrics)

app = gr.mount_gradio_app(app, io, path="/")
