from http import HTTPStatus

from fastapi.testclient import TestClient
from predicting_outcomes_in_heart_failure.app.main import app

client = TestClient(app)


def test_model_hyperparameters():
    response = client.get("/model/hyperparameters")
    assert response.status_code in (HTTPStatus.OK, HTTPStatus.NOT_FOUND)

    body = response.json()
    assert body["status-code"] in (HTTPStatus.OK.value, HTTPStatus.NOT_FOUND.value)

    if body["status-code"] == HTTPStatus.OK.value:
        assert "data" in body
        assert "hyperparameters" in body["data"]
    else:
        assert "data" in body
        assert "detail" in body["data"]
        assert body["data"]["detail"]


def test_model_metrics():
    response = client.get("/model/metrics")
    assert response.status_code in (HTTPStatus.OK, HTTPStatus.NOT_FOUND)

    body = response.json()

    if response.status_code == HTTPStatus.OK:
        assert body["status-code"] == HTTPStatus.OK.value
        data = body["data"]
        assert "model_path" in data
        assert "model_name" in data
        assert "variant" in data
        assert "metrics" in data

        metrics = data["metrics"]
        assert any(
            key in metrics for key in ("test_f1", "test_accuracy", "test_recall", "test_roc_auc")
        )
    else:
        assert body["status-code"] == HTTPStatus.NOT_FOUND.value
        assert "data" in body
        assert "detail" in body["data"]
