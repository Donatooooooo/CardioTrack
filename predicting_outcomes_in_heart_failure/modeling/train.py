from __future__ import annotations

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
    RANDOM_STATE,
    REPO_NAME,
    REPO_OWNER,
    REPORTS_DIR,
    SCORING,
    TARGET_COL,
    TRAIN_CSV,
)

REFIT = "f1"


def load_split(path: Path) -> pd.DataFrame:
    if not path.exists():
        logger.error(f"Missing split file: {path}. Run split_data.py first.")
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    logger.info(f"Loaded {path} (rows={len(df)}, cols={df.shape[1]})")
    return df


def get_model_and_grid(model_name: str):
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


def run_grid_search(estimator, param_grid, X_train, y_train, model_name):
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
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
    logger.info(f"Starting GridSearchCV for {model_name} …")
    grid.fit(X_train, y_train)

    logger.success("GridSearchCV completed.")
    logger.info(f"Best params ({REFIT}): {grid.best_params_}")
    logger.info(f"Best CV {REFIT}: {grid.best_score_:.4f}")

    cv_results = Path(REPORTS_DIR / model_name / "cv_results.json")
    df = pd.DataFrame(grid.cv_results_)
    df.to_csv(cv_results, index=False)

    mlflow.log_artifact(cv_results)
    return grid.best_estimator_, grid, grid.best_params_


def save_artifacts(model, grid, X_train, model_name) -> None:
    model_path = MODELS_DIR / f"{model_name}.joblib"

    joblib.dump(model, model_path)
    logger.success(f"Saved model → {model_path}")

    out = {
        "model_name": model_name,
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

    cv_params = Path(REPORTS_DIR / model_name / "cv_parameters.json")
    with open(REPORTS_DIR / cv_params, "w") as f:
        json.dump(out, f, indent=4)

    mlflow.log_artifact(cv_params)
    logger.success("Saved artifacts")


def train(model_name: str):
    if not mlflow.get_experiment_by_name(EXPERIMENT_NAME):
        mlflow.create_experiment(EXPERIMENT_NAME)
    mlflow.set_experiment(EXPERIMENT_NAME)

    logger.info(f"=== Training pipeline started (model={model_name}) ===")

    with mlflow.start_run(run_name=model_name):
        train_df = load_split(TRAIN_CSV)

        rawdata = mlflow.data.from_pandas(train_df, name=DATASET_NAME)
        mlflow.log_input(rawdata, context="training")

        X_train = train_df.drop(columns=[TARGET_COL])
        y_train = train_df[TARGET_COL].astype(int)

        estimator, param_grid = get_model_and_grid(model_name)
        mlflow.set_tag("estimator_name", estimator.__class__.__name__)

        model_dir = REPORTS_DIR / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        best_model, grid, params = run_grid_search(
            estimator, param_grid, X_train, y_train, model_name
        )
        mlflow.log_params(params)

        save_artifacts(best_model, grid, X_train, model_name)


def main():
    dagshub.init(repo_owner=REPO_OWNER, repo_name=REPO_NAME, mlflow=True)

    for model in ["logreg", "random_forest", "decision_tree"]:
        train(model)
    logger.success("Training pipeline completed.")


if __name__ == "__main__":
    main()
