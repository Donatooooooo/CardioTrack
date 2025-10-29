from __future__ import annotations
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, roc_auc_score, confusion_matrix
)
import matplotlib.pyplot as plt
from predicting_outcomes_in_heart_failure.config import (
    MODELS_DIR, REPORTS_DIR, FIGURES_DIR, TARGET_COL, 
    RANDOM_STATE, N_SPLITS, SCORING,TRAIN_CSV, TEST_CSV, 
)

REFIT = "f1"
MODEL_NAME = "random_forest"  # es: "decision_tree", "logreg", "random_forest"



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
        param_grid = {
            "criterion": ["gini", "entropy", "log_loss"],
            "max_depth": [None, 3, 5, 7, 9, 12],
            "min_samples_split": [2, 5, 10, 20],
            "min_samples_leaf": [1, 2, 4, 8],
            "max_features": [None, "sqrt", "log2"],
            "class_weight": [None, "balanced"],
            "ccp_alpha": [0.0, 0.001, 0.01],
        }
        return estimator, param_grid

    elif model_name == "logreg":
        from sklearn.linear_model import LogisticRegression
        estimator = LogisticRegression(max_iter=500, random_state=RANDOM_STATE)
        param_grid = {"C": [0.01, 0.1, 1, 10], "penalty": ["l2"], "class_weight": [None, "balanced"]}
        return estimator, param_grid

    elif model_name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        estimator = RandomForestClassifier(random_state=RANDOM_STATE)
        param_grid = {"n_estimators": [200, 400, 800], "max_depth": [None, 6, 12], "class_weight": [None, "balanced"]}
        return estimator, param_grid

    else:
        raise ValueError(f"Unknown model_name: {model_name}")


def run_grid_search(estimator, param_grid, X_train, y_train):
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
    logger.info(f"Starting GridSearchCV for {MODEL_NAME} …")
    grid.fit(X_train, y_train)
    logger.success("GridSearchCV completed.")
    logger.info(f"Best params ({REFIT}): {grid.best_params_}")
    logger.info(f"Best CV {REFIT}: {grid.best_score_:.4f}")
    return grid.best_estimator_, grid


def evaluate_metrics(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    results = {
        "test_f1": f1_score(y_test, y_pred, zero_division=0),
        "test_recall": recall_score(y_test, y_pred, zero_division=0),
        "test_accuracy": accuracy_score(y_test, y_pred),
    }
    if hasattr(model, "predict_proba"):
        try:
            y_prob = model.predict_proba(X_test)[:, 1]
            results["test_roc_auc"] = roc_auc_score(y_test, y_prob)
        except Exception as e:
            logger.warning(f"ROC AUC not computed: {e}")
    return results, y_pred


def save_confusion_matrix(y_true, y_pred, labels: list[str] | None = None) -> Path:
    cm = confusion_matrix(y_true, y_pred)
    fig_path = FIGURES_DIR / f"{MODEL_NAME}_confusion_matrix.png"

    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    if labels is None:
        labels = ["0", "1"]
    ticks = np.arange(len(labels))
    plt.xticks(ticks, labels)
    plt.yticks(ticks, labels)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()
    logger.success(f"Saved confusion matrix → {fig_path}")
    return fig_path


def save_artifacts(model, grid, X_train, y_test, y_pred, metrics: dict) -> None:
    model_path   = MODELS_DIR / f"{MODEL_NAME}.joblib"
    metrics_path = REPORTS_DIR / f"{MODEL_NAME}_metrics.json"

    joblib.dump(model, model_path)
    logger.success(f"Saved model → {model_path}")

    out = {
        "model_name": MODEL_NAME,
        "cv": {
            "refit": REFIT,
            "best_score": getattr(grid, "best_score_", None),
            "best_params": getattr(grid, "best_params_", None),
            "scoring": list(SCORING.keys()),
            "n_splits": N_SPLITS,
            "random_state": RANDOM_STATE,
        },
        "test": {
            **metrics,
            "y_true_counts": pd.Series(y_test).value_counts().to_dict(),
            "y_pred_counts": pd.Series(y_pred).value_counts().to_dict(),
        },
        "features": list(X_train.columns),
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    logger.success(f"Saved metrics → {metrics_path}")


def main():
    logger.info(f"=== Training pipeline started (model={MODEL_NAME}) ===")

    # 1) Load data
    train_df = load_split(TRAIN_CSV)
    test_df  = load_split(TEST_CSV)

    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL].astype(int)
    X_test  = test_df.drop(columns=[TARGET_COL])
    y_test  = test_df[TARGET_COL].astype(int)

    # 2) Model & Grid
    estimator, param_grid = get_model_and_grid(MODEL_NAME)

    # 3) GridSearchCV
    best_model, grid = run_grid_search(estimator, param_grid, X_train, y_train)

    # 4) Evaluate
    metrics, y_pred = evaluate_metrics(best_model, X_test, y_test)
    logger.info("Test set metrics:")
    for k in ["test_f1", "test_recall", "test_accuracy", "test_roc_auc"]:
        if k in metrics:
            logger.info(f"  - {k}: {metrics[k]:.4f}")

    save_confusion_matrix(y_test, y_pred, labels=["0", "1"])

    # 6) Save artifacts
    save_artifacts(best_model, grid, X_train, y_test, y_pred, metrics)

    logger.success("Training pipeline completed.")


if __name__ == "__main__":
    main()
