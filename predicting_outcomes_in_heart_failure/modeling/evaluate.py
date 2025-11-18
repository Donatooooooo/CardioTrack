import argparse
import os

import dagshub
import joblib
from loguru import logger
import mlflow
from mlflow.models.signature import infer_signature
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score

from predicting_outcomes_in_heart_failure.config import (
    DATASET_NAME,
    EXPERIMENT_NAME,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    REPO_NAME,
    REPO_OWNER,
    TARGET_COL,
    VALID_MODELS,
    VALID_VARIANTS,
)
from predicting_outcomes_in_heart_failure.modeling.train import load_split


def compute_metrics(model, X_test, y_test) -> dict:
    """Compute evaluation metrics (F1, recall, accuracy, ROC-AUC)."""
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


def evaluate_variant(variant: str, model_name: str | None = None):
    """Evaluate trained models for a given variant, optionally by model."""
    logger.info(f"=== Evaluation started (variant={variant}, model={model_name or 'ALL'}) ===")

    test_path = PROCESSED_DATA_DIR / variant / "test.csv"
    test_df = load_split(test_path)

    X_test = test_df.drop(columns=[TARGET_COL])
    y_test = test_df[TARGET_COL].astype(int)

    models_dir_variant = MODELS_DIR / variant
    if not models_dir_variant.exists():
        logger.warning(
            f"[{variant}] Models directory does not exist: {models_dir_variant} — skipping."
        )
        return

    experiment_name = f"{EXPERIMENT_NAME}_{variant}"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        logger.error(f"Experiment '{experiment_name}' not found.")
        return

    model_files = []
    if model_name is not None:
        model_files = [f"{model_name}.joblib"]
    else:
        model_files = [f for f in os.listdir(models_dir_variant) if f.endswith(".joblib")]

    for file in model_files:
        if not file.endswith(".joblib"):
            continue

        current_model_name = file.split(".joblib")[0]
        run_name = f"{current_model_name}_{variant}"
        logger.info(
            f"[{variant} | {current_model_name}] Looking for training run '{run_name}' in MLflow."
        )

        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"tags.mlflow.runName = '{run_name}'",
            order_by=["start_time DESC"],
            max_results=1,
        )

        if runs.empty:
            logger.warning(
                f"[{variant} | {current_model_name}]No matching MLflow run found — skipping."
            )
            continue

        tracked_id = runs.loc[0, "run_id"]

        with mlflow.start_run(run_id=tracked_id):
            rawdata = mlflow.data.from_pandas(test_df, name=f"{DATASET_NAME}_{variant}_test")
            mlflow.log_input(rawdata, context="testing")

            model_path = models_dir_variant / file
            model = joblib.load(model_path)

            metrics, _ = compute_metrics(model, X_test, y_test)
            mlflow.log_metrics(metrics)

            logger.info(f"[{variant} | {current_model_name}] Test set metrics:")
            for k in ["test_f1", "test_recall", "test_accuracy", "test_roc_auc"]:
                if k in metrics:
                    logger.info(f"  - {k}: {metrics[k]:.4f}")

            if (
                metrics.get("test_f1", 0.0) >= 0.80
                and metrics.get("test_recall", 0.0) >= 0.80
                and metrics.get("test_accuracy", 0.0) >= 0.80
                and metrics.get("test_roc_auc", 0.0) >= 0.85
            ):
                signature = infer_signature(X_test, model.predict(X_test))
                registered_name = f"{current_model_name}_{variant}"
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="Model_Info",
                    signature=signature,
                    input_example=X_test,
                    registered_model_name=registered_name,
                )
                logger.success(
                    f"[{variant} | {current_model_name}] "
                    f"Model promoted and registered as '{registered_name}'."
                )

    logger.success(
        f"=== Evaluation completed (variant={variant}, model={model_name or 'ALL'}) ==="
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
        required=False,
        help=(
            "Specific model to evaluate (logreg, random_forest, decision_tree)."
            " If omitted, evaluate all models."
        ),
    )
    args = parser.parse_args()

    dagshub.init(repo_owner=REPO_OWNER, repo_name=REPO_NAME, mlflow=True)
    evaluate_variant(args.variant, args.model)


if __name__ == "__main__":
    main()
