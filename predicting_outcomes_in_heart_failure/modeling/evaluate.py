from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, roc_auc_score
)
from loguru import logger
from predicting_outcomes_in_heart_failure.config import MODELS_DIR, TARGET_COL, TEST_CSV, EXPERIMENT_NAME
from train import load_split
import os, joblib
import dagshub, mlflow


def compute_metrics(model, X_test, y_test) -> dict:
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

def evaluate():
    test_df  = load_split(TEST_CSV)

    X_test  = test_df.drop(columns=[TARGET_COL])
    y_test  = test_df[TARGET_COL].astype(int)

    for file in os.listdir(MODELS_DIR):
        if file.endswith(".joblib"):
            
            model_name = file.split(".joblib")[0]
            experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string=f"tags.mlflow.runName = '{model_name}'",
                order_by=["start_time DESC"],
                max_results=1
            )
            
            tracked_id = runs.loc[0, "run_id"]
            with mlflow.start_run(run_id = tracked_id):
                model_path = os.path.join(MODELS_DIR, file)
                model = joblib.load(model_path)
                metrics, _ = compute_metrics(model, X_test, y_test)
                
                mlflow.log_metrics(metrics)
                logger.info(f"{model_name} - Test set metrics:")
                for k in ["test_f1", "test_recall", "test_accuracy", "test_roc_auc"]:
                    if k in metrics:
                        logger.info(f"  - {k}: {metrics[k]:.4f}")
                    
if __name__ == "__main__":
    dagshub.init(repo_owner='donatooooooo', repo_name='MLflow_Server', mlflow=True)
    evaluate()
    logger.success("Evaluation completed.")