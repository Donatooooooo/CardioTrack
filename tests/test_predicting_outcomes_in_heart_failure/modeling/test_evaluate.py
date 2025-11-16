import pytest
import numpy as np
import pandas as pd
from types import SimpleNamespace
import types

from predicting_outcomes_in_heart_failure.modeling import evaluate
from predicting_outcomes_in_heart_failure.config import TARGET_COL


@pytest.mark.parametrize(
    "model_fixture",
    ["logreg_model", "decision_tree_model", "random_forest_model"],
)
def test_compute_metrics_all_models(request, model_fixture, definition_X_test_and_y_test):
    """
    Verifica che compute_metrics funzioni correttamente per tutti i modelli
    salvati nella variant 'all'.
    """

    X_test, y_test = definition_X_test_and_y_test

    model = request.getfixturevalue(model_fixture)


    results, y_pred = evaluate.compute_metrics(model, X_test, y_test)

    # ---- ASSERTIONS ----
    assert isinstance(results, dict)
    assert len(y_pred) == len(y_test)

    for key in ["test_f1", "test_recall", "test_accuracy"]:
        assert key in results

    if hasattr(model, "predict_proba"):
        assert "test_roc_auc" in results
        assert 0.0 <= results["test_roc_auc"] <= 1.0
    else:
        assert "test_roc_auc" not in results    

class DummyModelNoProba:
    def __init__(self, y_pred):
        self._y_pred = np.array(y_pred)

    def predict(self, X):
        return self._y_pred


def test_compute_metrics_without_predict_proba():
    X_test = np.array([[0], [1], [2], [3]])
    y_test = np.array([0, 1, 0, 1])
    model = DummyModelNoProba(y_pred=[0, 1, 0, 0])

    results, y_pred = evaluate.compute_metrics(model, X_test, y_test)

    assert isinstance(results, dict)
    assert len(y_pred) == len(y_test)

    assert "test_f1" in results
    assert "test_recall" in results
    assert "test_accuracy" in results
    assert "test_roc_auc" not in results     




def test_evaluate_variant_skips_when_models_dir_missing(
    monkeypatch,
    tmp_path,
    sample_raw_df_two_rows,
):
    """
    If MODELS_DIR / <variant> does NOT exist,
    evaluate_variant should:

      - return early
      - NOT call any MLflow APIs (e.g. get_experiment_by_name)
    """
    variant = "all"

    def fake_load_split(path):
        df = sample_raw_df_two_rows.copy()
        df[TARGET_COL] = [0, 1]
        return df

    monkeypatch.setattr(evaluate, "load_split", fake_load_split)


    models_root = tmp_path / "models"
    models_root.mkdir()  # root exists, but models_root / "all" does not
    monkeypatch.setattr(evaluate, "MODELS_DIR", models_root)

    # 3) Dummy mlflow: we just count calls to get_experiment_by_name
    class DummyMlflow:
        called_get_experiment = 0

        @staticmethod
        def get_experiment_by_name(name):
            DummyMlflow.called_get_experiment += 1
            return None

    monkeypatch.setattr(evaluate, "mlflow", DummyMlflow)

    evaluate.evaluate_variant(variant=variant, model_name=None)


    assert DummyMlflow.called_get_experiment == 0





def test_evaluate_variant_skips_model_when_no_mlflow_run(
    monkeypatch,
    tmp_path,
    processed_df,
    dummy_logger,
    mlflow_no_runs,
):
    """
    Scenario:
      - MODELS_DIR / 'all' exists and contains a .joblib model file
      - mlflow.search_runs(...) returns an empty DataFrame

    We check that:
      - mlflow.search_runs is called
      - compute_metrics is NOT called
      - a warning 'No matching MLflow run found — skipping.' is logged
    """
    variant = "all"
    model_name = "random_forest"

    # Fake load_split → always returns processed_df
    def fake_load_split(path):
        return processed_df.copy()

    monkeypatch.setattr(evaluate, "load_split", fake_load_split)

    # MODELS_DIR / <variant> with a fake .joblib file
    models_root = tmp_path / "models"
    models_variant_dir = models_root / variant
    models_variant_dir.mkdir(parents=True)

    model_filename = f"{model_name}.joblib"
    model_path = models_variant_dir / model_filename
    model_path.write_bytes(b"fake-binary-content")

    monkeypatch.setattr(evaluate, "MODELS_DIR", models_root)

    # compute_metrics should NOT be called
    compute_metrics_calls = {"count": 0}

    def fake_compute_metrics(model, X, y):
        compute_metrics_calls["count"] += 1
        return {"test_f1": 0.5}, None

    monkeypatch.setattr(evaluate, "compute_metrics", fake_compute_metrics)

    # ACT
    evaluate.evaluate_variant(variant=variant, model_name=model_name)

    # ASSERT
    assert mlflow_no_runs.called_search_runs == 1
    assert compute_metrics_calls["count"] == 0
    assert any(
        "No matching MLflow run found — skipping." in msg
        for msg in dummy_logger.warnings
    )


def test_evaluate_variant_returns_when_experiment_missing(
    monkeypatch,
    tmp_path,
    sample_raw_df_two_rows,
    dummy_logger,
    mlflow_experiment_missing,
):
    """
    If mlflow.get_experiment_by_name(...) returns None,
    evaluate_variant should:

      - log an error
      - return early
      - NOT call mlflow.search_runs
    """
    variant = "all"

    def fake_load_split(path):
        df = sample_raw_df_two_rows.copy()
        df[TARGET_COL] = [0, 1]
        return df

    monkeypatch.setattr(evaluate, "load_split", fake_load_split)

    models_root = tmp_path / "models"
    models_variant_dir = models_root / variant
    models_variant_dir.mkdir(parents=True)
    monkeypatch.setattr(evaluate, "MODELS_DIR", models_root)

    evaluate.evaluate_variant(variant=variant, model_name=None)

    assert mlflow_experiment_missing.called_get_experiment == 1
    assert mlflow_experiment_missing.called_search_runs == 0
    assert any("Experiment" in msg for msg in dummy_logger.errors)


def test_evaluate_variant_happy_path_without_promotion(
    monkeypatch,
    tmp_path,
    sample_raw_df_two_rows,
    dummy_logger,
):
    """
    Minimal happy path:
      - MODELS_DIR / 'all' exists and contains a .joblib model
      - MLflow experiment and run are found
      - compute_metrics is called
      - metrics < thresholds → model is NOT registered
    """
    variant = "all"
    model_name = "random_forest"

    def fake_load_split(path):
        df = sample_raw_df_two_rows.copy()
        df[TARGET_COL] = [0, 1]
        return df

    monkeypatch.setattr(evaluate, "load_split", fake_load_split)

    # MODELS_DIR / variant with a .joblib file
    models_root = tmp_path / "models"
    models_variant_dir = models_root / variant
    models_variant_dir.mkdir(parents=True)
    model_path = models_variant_dir / f"{model_name}.joblib"
    model_path.write_bytes(b"fake-binary-content")
    monkeypatch.setattr(evaluate, "MODELS_DIR", models_root)

    # Fake model + joblib.load
    class DummyModel:
        def predict(self, X):
            return [0] * len(X)

    def fake_joblib_load(path):
        assert path == model_path
        return DummyModel()

    monkeypatch.setattr(
        evaluate, "joblib", types.SimpleNamespace(load=fake_joblib_load)
    )

    # compute_metrics called once, returning low metrics
    compute_metrics_calls = {"count": 0}

    def fake_compute_metrics(model, X, y):
        compute_metrics_calls["count"] += 1
        return {
            "test_f1": 0.5,
            "test_recall": 0.5,
            "test_accuracy": 0.5,
            "test_roc_auc": 0.5,
        }, None

    monkeypatch.setattr(evaluate, "compute_metrics", fake_compute_metrics)

    # Fake MLflow: experiment + run exist, but we track log_model calls
    class DummyRunCtx:
        def __init__(self, run_id):
            self.run_id = run_id

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class DummyMlflow:
        called_search_runs = 0
        started_runs = []
        log_model_calls = 0

        class sklearn:
            @staticmethod
            def log_model(*args, **kwargs):
                DummyMlflow.log_model_calls += 1

        class data:
            @staticmethod
            def from_pandas(*args, **kwargs):
                return "dummy-dataset"

        @staticmethod
        def log_input(*args, **kwargs):
            pass

        @staticmethod
        def log_metrics(*args, **kwargs):
            pass

        @staticmethod
        def get_experiment_by_name(name):
            return SimpleNamespace(experiment_id="exp-123")

        @staticmethod
        def search_runs(experiment_ids, filter_string, order_by, max_results):
            DummyMlflow.called_search_runs += 1
            return pd.DataFrame([{"run_id": "run-123"}])

        @staticmethod
        def start_run(run_id):
            DummyMlflow.started_runs.append(run_id)
            return DummyRunCtx(run_id)

    monkeypatch.setattr(evaluate, "mlflow", DummyMlflow)

    
    evaluate.evaluate_variant(variant=variant, model_name=model_name)

    # ASSERTIONS
    assert compute_metrics_calls["count"] == 1
    assert DummyMlflow.called_search_runs == 1
    assert DummyMlflow.started_runs == ["run-123"]
    # metrics below thresholds → no promotion → no log_model
    assert DummyMlflow.log_model_calls == 0
