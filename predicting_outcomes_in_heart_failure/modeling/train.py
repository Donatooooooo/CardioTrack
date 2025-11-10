from __future__ import annotations

import argparse
import json
from pathlib import Path

import dagshub
import joblib
from loguru import logger
import mlflow
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from predicting_outcomes_in_heart_failure.config import (
    CONFIG_DT,
    CONFIG_LR,
    CONFIG_RF,
    DATASET_NAME,
    EXPERIMENT_NAME,
    MODELS_DIR,
    N_SPLITS,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    REPO_NAME,
    REPO_OWNER,
    REPORTS_DIR,
    SCORING,
    TARGET_COL,
)

REFIT = "f1"
VALID_MODELS = ["logreg", "random_forest", "decision_tree"]
VALID_VARIANTS = ["all", "female", "male", "nosex"]


def load_split(path: Path) -> pd.DataFrame:
    if not path.exists():
        logger.error(f"Missing split file: {path}. Run split_data.py first.")
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    logger.info(f"Loaded {path} (rows={len(df)}, cols={df.shape[1]})")
    return df


def get_model_and_grid(model_name: str):
    """Return estimator and parameter grid for the selected model."""
    if model_name == "decision_tree":
        from sklearn.tree import DecisionTreeClassifier

        estimator = DecisionTreeClassifier(random_state=RANDOM_STATE)
        param_grid = CONFIG_DT
        return estimator, param_grid

    elif model_name == "logreg":
        from sklearn.linear_model import LogisticRegression

        estimator = LogisticRegression(max_iter=500, random_state=RANDOM_STATE)
        param_grid = CONFIG_LR
        return estimator, param_grid

    elif model_name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier

        estimator = RandomForestClassifier(random_state=RANDOM_STATE)
        param_grid = CONFIG_RF
        return estimator, param_grid

    else:
        raise ValueError(f"Unknown model_name: {model_name}")


def run_grid_search(
    estimator,
    param_grid,
    X_train,
    y_train,
    model_name: str,
    variant: str,
    reports_dir: Path,
):
    """Run GridSearchCV for the specified model and log CV results."""
    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    grid = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring=SCORING,
        refit=REFIT,
        cv=cv,
        n_jobs=-1,
        verbose=1,
        return_train_score=True,
    )

    logger.info(f"[{variant} | {model_name}] Starting GridSearchCV …")
    grid.fit(X_train, y_train)

    logger.success(f"[{variant} | {model_name}] GridSearchCV completed.")
    logger.info(
        f"[{variant} | {model_name}] Best params ({REFIT}): {grid.best_params_}"
    )
    logger.info(f"[{variant} | {model_name}] Best CV {REFIT}: {grid.best_score_:.4f}")

    cv_results_path = reports_dir / "cv_results.csv"
    df = pd.DataFrame(grid.cv_results_)
    df.to_csv(cv_results_path, index=False)

    mlflow.log_artifact(str(cv_results_path))
    return grid.best_estimator_, grid, grid.best_params_


def save_artifacts(
    model,
    grid,
    X_train,
    model_name: str,
    variant: str,
    model_dir: Path,
    reports_dir: Path,
) -> None:
    """Save model, parameters, and metadata to disk and MLflow."""
    model_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / f"{model_name}.joblib"
    joblib.dump(model, model_path)
    logger.success(f"[{variant} | {model_name}] Saved model → {model_path}")

    out = {
        "model_name": model_name,
        "data_variant": variant,
        "cv": {
            "refit": REFIT,
            "best_score": getattr(grid, "best_score_", None),
            "best_params": getattr(grid, "best_params_", None),
            "scoring": list(SCORING.keys()),
            "n_splits": N_SPLITS,
            "random_state": RANDOM_STATE,
        },
        "features": list(X_train.columns),
    }

    cv_params_path = reports_dir / "cv_parameters.json"
    with open(cv_params_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=4)

    mlflow.log_artifact(str(cv_params_path))
    logger.success(f"[{variant} | {model_name}] Saved artifacts.")


def train(model_name: str, variant: str):
    """Train a model for a specific dataset variant and log results to MLflow."""
    if not mlflow.get_experiment_by_name(EXPERIMENT_NAME):
        mlflow.create_experiment(EXPERIMENT_NAME)
    mlflow.set_experiment(EXPERIMENT_NAME)

    train_path = PROCESSED_DATA_DIR / variant / "train.csv"
    run_name = f"{model_name}_{variant}"

    logger.info(f"=== Training started (model={model_name}, variant={variant}) ===")

    with mlflow.start_run(run_name=run_name):
        train_df = load_split(train_path)

        rawdata = mlflow.data.from_pandas(train_df, name=f"{DATASET_NAME}_{variant}")
        mlflow.log_input(rawdata, context="training")

        X_train = train_df.drop(columns=[TARGET_COL])
        y_train = train_df[TARGET_COL].astype(int)

        estimator, param_grid = get_model_and_grid(model_name)
        mlflow.set_tag("estimator_name", estimator.__class__.__name__)
        mlflow.set_tag("data_variant", variant)
        mlflow.log_param("data_variant", variant)

        model_dir = MODELS_DIR / variant
        reports_dir = REPORTS_DIR / variant / model_name
        reports_dir.mkdir(parents=True, exist_ok=True)

        best_model, grid, params = run_grid_search(
            estimator,
            param_grid,
            X_train,
            y_train,
            model_name=model_name,
            variant=variant,
            reports_dir=reports_dir,
        )
        mlflow.log_params(params)

        save_artifacts(
            best_model,
            grid,
            X_train,
            model_name=model_name,
            variant=variant,
            model_dir=model_dir,
            reports_dir=reports_dir,
        )

    logger.success(
        f"=== Training completed (model={model_name}, variant={variant}) ==="
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        type=str,
        choices=VALID_VARIANTS,
        required=True,
        help="Data variant to use: all, female, male, or nosex.",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=VALID_MODELS,
        required=True,
        help="Model to train: logreg, random_forest, or decision_tree.",
    )
    args = parser.parse_args()

    dagshub.init(repo_owner=REPO_OWNER, repo_name=REPO_NAME, mlflow=True)
    train(args.model, args.variant)


if __name__ == "__main__":
    main()
