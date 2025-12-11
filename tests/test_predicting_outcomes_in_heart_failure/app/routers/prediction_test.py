from http import HTTPStatus

from fastapi.testclient import TestClient
import joblib
from predicting_outcomes_in_heart_failure.app.main import app
from predicting_outcomes_in_heart_failure.config import MODEL_PATH
import pytest

client = TestClient(app)

SINGLE_SAMPLE = {
    "Age": 45,
    "ChestPainType": "TA",
    "RestingBP": 120,
    "Cholesterol": 200,
    "FastingBS": 0,
    "RestingECG": "Normal",
    "MaxHR": 150,
    "ExerciseAngina": "N",
    "Oldpeak": 0.0,
    "ST_Slope": "Up",
}

BATCH_PAYLOAD = [
    {
        "Age": 45,
        "ChestPainType": "TA",
        "RestingBP": 120,
        "Cholesterol": 200,
        "FastingBS": 0,
        "RestingECG": "Normal",
        "MaxHR": 150,
        "ExerciseAngina": "N",
        "Oldpeak": 0.0,
        "ST_Slope": "Up",
    },
    {
        "Age": 67,
        "ChestPainType": "ASY",
        "RestingBP": 160,
        "Cholesterol": 280,
        "FastingBS": 1,
        "RestingECG": "LVH",
        "MaxHR": 105,
        "ExerciseAngina": "Y",
        "Oldpeak": 2.5,
        "ST_Slope": "Flat",
    },
]


@pytest.fixture
def ensure_model_loaded():
    """
    Fixture that ensures app.state.model is loaded
    before running the test.
    """
    if not hasattr(app.state, "model") or app.state.model is None:
        app.state.model = joblib.load(MODEL_PATH)
    yield


def test_predictions_single_ok(ensure_model_loaded):
    """Test the /predictions endpoint with a single valid example."""
    response = client.post("/predictions", json=SINGLE_SAMPLE)

    assert response.status_code == HTTPStatus.OK

    body = response.json()
    assert body["status-code"] == HTTPStatus.OK.value
    assert body["method"] == "POST"
    assert "timestamp" in body
    assert "url" in body
    assert "data" in body

    data = body["data"]

    # input must match payload
    assert data["input"] == SINGLE_SAMPLE

    # prediction must be an integer 0/1
    assert isinstance(data["prediction"], int)
    assert data["prediction"] in (0, 1)


def test_predict_batch_two_samples_ok(ensure_model_loaded):
    """Test /predict-batch with two examples."""
    response = client.post("/batch-predictions", json=BATCH_PAYLOAD)

    assert response.status_code == HTTPStatus.OK

    body = response.json()

    assert body["status-code"] == HTTPStatus.OK.value
    assert body["method"] == "POST"
    assert "data" in body

    data = body["data"]

    # batch_size coerente con la lunghezza del payload
    assert data["batch_size"] == len(BATCH_PAYLOAD)

    results = data["results"]
    assert isinstance(results, list)
    assert len(results) == len(BATCH_PAYLOAD)

    # Check every element of the batch
    for idx, item in enumerate(results):
        assert item["index"] == idx
        assert item["input"] == BATCH_PAYLOAD[idx]
        assert isinstance(item["prediction"], int)
        assert item["prediction"] in (0, 1)


def test_explanations_single_ok(ensure_model_loaded):
    """Test /explanations with a single example."""
    response = client.post("/explanations", json=SINGLE_SAMPLE)

    assert response.status_code == HTTPStatus.OK

    body = response.json()

    assert body["status-code"] == HTTPStatus.OK.value
    assert body["method"] == "POST"
    assert "data" in body

    data = body["data"]
    assert data["input"] == SINGLE_SAMPLE

    if "explanations" in data:
        assert isinstance(data["explanations"], list)

    if "explanation_plot_url" in data:
        assert isinstance(data["explanation_plot_url"], str)
        assert data["explanation_plot_url"].startswith("/figures/")


def test_predictions_model_not_loaded():
    """
    Verify that when the model is missing from app.state, the endpoint returns
    a 503 status-code inside the response payload while keeping HTTP 200.
    """
    app.state.model = None

    response = client.post("/predictions", json=SINGLE_SAMPLE)
    assert response.status_code == HTTPStatus.OK

    body = response.json()
    assert body["status-code"] == HTTPStatus.SERVICE_UNAVAILABLE.value
    assert body["data"]["detail"] == "Model is not loaded."


def test_predictions_invalid_type():
    """
    Ensure that invalid field types (e.g., Age as string) trigger FastAPI/Pydantic
    validation and return HTTP 422 instead of reaching the endpoint logic.
    """
    bad_payload = SINGLE_SAMPLE.copy()
    bad_payload["Age"] = "not_a_number"

    response = client.post("/predictions", json=bad_payload)
    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY

    body = response.json()
    assert "status-code" not in body
    assert "data" not in body
    assert "detail" in body

    errors = body["detail"]
    assert isinstance(errors, list)
    assert len(errors) > 0
