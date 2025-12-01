from http import HTTPStatus

from fastapi.testclient import TestClient
from predicting_outcomes_in_heart_failure.app.main import app

client = TestClient(app)


def test_index_root():
    response = client.get("/")
    assert response.status_code == HTTPStatus.OK

    body = response.json()
    assert body["status-code"] == HTTPStatus.OK.value
    assert body["method"] == "GET"
    assert "timestamp" in body
    assert "data" in body
    assert body["data"]["message"] == "Welcome to Heart Failure Predictor!"
