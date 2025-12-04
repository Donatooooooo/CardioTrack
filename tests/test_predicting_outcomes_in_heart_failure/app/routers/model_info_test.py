from http import HTTPStatus

from fastapi.testclient import TestClient
from predicting_outcomes_in_heart_failure.app.main import app

client = TestClient(app)


def test_model_hyperparameters():
    response = client.get("/model/hyperparameters")
    assert response.status_code in (HTTPStatus.OK, HTTPStatus.NOT_FOUND)

    body = response.json()

    if response.status_code == HTTPStatus.OK:
        assert body["status-code"] == HTTPStatus.OK.value
        data = body["data"]
        assert "model_path" in data
        assert "hyperparameters" in data

        hyper = data["hyperparameters"]
        assert "model_name" in hyper
        assert "data_variant" in hyper
        assert "cv" in hyper
        assert "features" in hyper
    else:
        assert body["status-code"] == HTTPStatus.NOT_FOUND.value
        assert "data" in body
        assert "detail" in body["data"]


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
