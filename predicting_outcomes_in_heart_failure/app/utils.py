from datetime import datetime
from functools import wraps

from fastapi import Request
import gradio as gr
from loguru import logger


def construct_response(f):
    """Construct a JSON response for an endpoint's results."""

    @wraps(f)
    def wrap(request: Request, *args, **kwargs):
        result = f(request, *args, **kwargs)
        response = {
            "message": result["message"],
            "method": request.method,
            "status-code": result["status-code"],
            "timestamp": datetime.now().isoformat(),
            "url": request.url._url,
        }
        if "data" in result:
            response["data"] = result["data"]
        return response

    return wrap


def get_model_from_state(request: Request):
    """Retrieve the model from the app state."""
    model = getattr(request.app.state, "model", None)
    if model is None:
        logger.error("Model not loaded in app.state.model")
    return model


def load_page(io, fn):
    content = gr.Markdown("Loading...")

    io.load(fn=fn, inputs=None, outputs=content)
    return io


def update_patient_index_choices(df):
    """Populate the dropdown with valid patient indices from the batch results."""
    import gradio as gr

    if df is None:
        return gr.update(choices=[], value=None)

    try:
        indices = list(df["Patients's index"].astype(int))
    except Exception:
        return gr.update(choices=[], value=None)

    if not indices:
        return gr.update(choices=[], value=None)

    return gr.update(choices=indices, value=indices[0])
