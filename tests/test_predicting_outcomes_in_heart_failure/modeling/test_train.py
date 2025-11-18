import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from predicting_outcomes_in_heart_failure.modeling import train
import pytest

# =========================
#  load_split
# =========================


def test_load_split_ok(tmp_path: Path):
    """
    Test that `load_split` correctly reads a CSV file.

    The test creates a temporary CSV with known columns and values,
    loads it using `load_split`, and verifies that:
      - the resulting DataFrame is not empty,
      - the column names match the original ones,
      - the number of rows is preserved.
    """
    path = tmp_path / "train.csv"
    df_orig = pd.DataFrame(
        {
            "feat1": [1, 2, 3],
            "feat2": [0.1, 0.2, 0.3],
            train.TARGET_COL: [0, 1, 0],
        }
    )
    df_orig.to_csv(path, index=False)

    df_loaded = train.load_split(path)

    assert not df_loaded.empty
    assert list(df_loaded.columns) == list(df_orig.columns)
    assert len(df_loaded) == len(df_orig)


def test_load_split_missing_file_raises(tmp_path: Path):
    """
    Test that `load_split` raises FileNotFoundError when the CSV file does not exist.
    """
    path = tmp_path / "missing.csv"
    assert not path.exists()

    with pytest.raises(FileNotFoundError):
        train.load_split(path)


# =========================
#  apply_random_oversampling
# =========================


def test_apply_random_oversampling_balances_classes():
    """
    Test that `apply_random_oversampling` correctly balances class distribution.

    The test creates an imbalanced dataset, applies random oversampling,
    and verifies that:
      - both classes appear with the same frequency after resampling,
      - the feature columns remain unchanged,
      - the number of samples in X and y is consistent.
    """
    X = pd.DataFrame(
        {
            "feat1": [1, 2, 3, 4],
            "feat2": [10, 20, 30, 40],
        }
    )
    y = pd.Series([0, 1, 1, 1], name=train.TARGET_COL)

    X_res, y_res = train.apply_random_oversampling(X, y, model_name="logreg", variant="all")

    counts = y_res.value_counts().to_dict()

    assert counts[0] == counts[1]
    assert list(X_res.columns) == list(X.columns)
    assert len(X_res) == len(y_res)


# =========================
#  get_model_and_grid
# =========================


@pytest.mark.parametrize(
    "model_name, expected_class",
    [
        ("decision_tree", "DecisionTreeClassifier"),
        ("logreg", "LogisticRegression"),
        ("random_forest", "RandomForestClassifier"),
    ],
)
def test_get_model_and_grid_valid(model_name, expected_class):
    """
    Test that `get_model_and_grid` returns the correct estimator and parameter grid.

    For each supported model name, the test verifies that:
      - the returned estimator matches the expected class,
      - the parameter grid is a dictionary,
      - the grid contains at least one hyperparameter.
    """
    estimator, param_grid = train.get_model_and_grid(model_name)
    assert estimator.__class__.__name__ == expected_class
    assert isinstance(param_grid, dict)
    assert len(param_grid) > 0


def test_get_model_and_grid_invalid():
    """
    Test that `get_model_and_grid` raises a ValueError for unsupported model names.
    """
    with pytest.raises(ValueError):
        train.get_model_and_grid("unknown_model")


# =========================
#  run_grid_search
# =========================


def test_run_grid_search_creates_cv_results(tmp_path: Path, monkeypatch):
    """
    Test that `run_grid_search` executes a full GridSearchCV workflow
    and produces the expected outputs.

    The test:
      - patches `N_SPLITS` to speed up cross-validation,
      - disables `mlflow.log_artifact` during the run,
      - runs grid search on a small synthetic dataset,
      - verifies that the `cv_results.csv` file is created,
      - checks that the returned best model is usable,
      - ensures the best hyperparameters include the parameter `C`.
    """
    from sklearn.linear_model import LogisticRegression

    monkeypatch.setattr(train, "N_SPLITS", 2, raising=False)
    monkeypatch.setattr(train.mlflow, "log_artifact", lambda *_, **__: None)

    estimator = LogisticRegression(max_iter=200)
    param_grid = {"C": [0.1, 1.0]}

    X_train = pd.DataFrame(
        np.random.randn(20, 2),
        columns=["feat1", "feat2"],
    )
    y_train = pd.Series([0] * 10 + [1] * 10, name=train.TARGET_COL)

    reports_dir = tmp_path / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    best_model, grid, best_params = train.run_grid_search(
        estimator=estimator,
        param_grid=param_grid,
        X_train=X_train,
        y_train=y_train,
        model_name="logreg",
        variant="all",
        reports_dir=reports_dir,
    )

    cv_results_path = reports_dir / "cv_results.csv"

    assert cv_results_path.exists()
    assert hasattr(best_model, "predict")
    assert isinstance(best_params, dict)
    assert "C" in best_params


# =========================
#  save_artifacts
# =========================


def test_save_artifacts_writes_files(tmp_path: Path, monkeypatch):
    """
    Test that `save_artifacts` correctly writes the model and CV metadata files.

    The test:
      - disables `mlflow.log_artifact` to avoid side effects,
      - uses a dummy model and dummy grid search results,
      - calls `save_artifacts` with temporary output directories,
      - verifies that the serialized model file and `cv_parameters.json`
        are created,
      - checks that the JSON file contains the expected metadata such as
        model name, variant, refit metric, and feature list.
    """
    from sklearn.linear_model import LogisticRegression

    monkeypatch.setattr(train.mlflow, "log_artifact", lambda *_, **__: None)

    model = LogisticRegression()
    model_dir = tmp_path / "models"
    reports_dir = tmp_path / "reports"

    class DummyGrid:
        best_score_ = 0.9
        best_params_ = {"C": 1.0}

    grid = DummyGrid()

    X_train = pd.DataFrame(
        {"feat1": [1, 2], "feat2": [3, 4]},
    )

    train.save_artifacts(
        model=model,
        grid=grid,
        X_train=X_train,
        model_name="logreg",
        variant="all",
        model_dir=model_dir,
        reports_dir=reports_dir,
    )

    model_path = model_dir / "logreg.joblib"
    cv_params_path = reports_dir / "cv_parameters.json"

    assert model_path.exists()
    assert cv_params_path.exists()

    with open(cv_params_path, encoding="utf-8") as f:
        data = json.load(f)

    assert data["model_name"] == "logreg"
    assert data["data_variant"] == "all"
    assert data["cv"]["refit"] == train.REFIT
    assert data["features"] == list(X_train.columns)


# =========================
#  train
# =========================


class _DummyRun:
    """
    Minimal dummy context manager used to mock an MLflow run.
    """

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _DummyData:
    """
    Dummy object used to replace `mlflow.data`.
    """

    @staticmethod
    def from_pandas(df, name: str):
        return df


def test_train_end_to_end_small(tmp_path: Path, monkeypatch):
    """
    End-to-end style test ensuring that `train()` runs successfully and
    produces the expected model artifact.

    The test:
      - patches paths,
      - overrides MLflow functions to avoid external side effects,
      - creates a small synthetic dataset stored as `train.csv`,
      - calls `train()` with the "logreg" model,
      - verifies that the trained model file is saved in the expected location.
    """

    processed_dir = tmp_path / "processed"
    models_dir = tmp_path / "models"
    reports_dir = tmp_path / "reports"

    monkeypatch.setattr(train, "PROCESSED_DATA_DIR", processed_dir, raising=False)
    monkeypatch.setattr(train, "MODELS_DIR", models_dir, raising=False)
    monkeypatch.setattr(train, "REPORTS_DIR", reports_dir, raising=False)
    monkeypatch.setattr(train, "N_SPLITS", 2, raising=False)

    variant = "all"
    split_dir = processed_dir / variant
    split_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(
        {
            "feat1": [0, 1, 0, 1],
            "feat2": [1, 1, 0, 0],
            train.TARGET_COL: [0, 1, 0, 1],
        }
    )
    (split_dir / "train.csv").write_text(df.to_csv(index=False))

    monkeypatch.setattr(train.mlflow, "get_experiment_by_name", lambda name: None)
    monkeypatch.setattr(train.mlflow, "create_experiment", lambda name: None)
    monkeypatch.setattr(train.mlflow, "set_experiment", lambda name: None)
    monkeypatch.setattr(train.mlflow, "start_run", lambda **kwargs: _DummyRun())
    monkeypatch.setattr(train.mlflow, "log_input", lambda *_, **__: None)
    monkeypatch.setattr(train.mlflow, "set_tag", lambda *_, **__: None)
    monkeypatch.setattr(train.mlflow, "log_param", lambda *_, **__: None)
    monkeypatch.setattr(train.mlflow, "log_params", lambda *_, **__: None)
    monkeypatch.setattr(train.mlflow, "log_artifact", lambda *_, **__: None)
    monkeypatch.setattr(train.mlflow, "data", _DummyData)

    train.train(model_name="logreg", variant=variant)
    model_path = models_dir / variant / "logreg.joblib"

    assert model_path.exists()


# =========================
#  main
# =========================


def test_main_parses_args_and_calls_train(monkeypatch):
    """
    Test that the `main` function correctly parses CLI arguments and
    forwards them to the `train` function.

    The test:
      - replaces `train.train` with a fake function to capture calls,
      - mocks `sys.argv` to simulate command-line input,
      - ensures that `main()` extracts the arguments properly,
      - verifies that the correct `model_name` and `variant`
        are passed to `train`.
    """
    called = {}

    def fake_train(model_name, variant):
        called["model_name"] = model_name
        called["variant"] = variant

    monkeypatch.setattr(train, "train", fake_train)
    monkeypatch.setattr(train.dagshub, "init", lambda *_, **__: None)

    test_args = ["prog", "--variant", "all", "--model", "logreg"]
    monkeypatch.setattr(sys, "argv", test_args)

    train.main()

    assert called["model_name"] == "logreg"
    assert called["variant"] == "all"
