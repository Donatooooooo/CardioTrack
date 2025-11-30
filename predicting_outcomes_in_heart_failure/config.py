from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

load_dotenv()

# -------------------
# Experiment settings
# -------------------
VALID_VARIANTS = ["all", "female", "male", "nosex"]
VALID_MODELS = ["logreg", "random_forest", "decision_tree"]
EXPERIMENT_NAME = "Heart_Failure_Prediction"
DATASET_NAME = "fedesoriano/heart-failure-prediction"
TARGET_COL = "HeartDisease"
RANDOM_STATE = 42
TEST_SIZE = 0.30
N_SPLITS = 5
SCORING = {
    "accuracy": "accuracy",
    "f1": "f1",
    "recall": "recall",
    "roc_auc": "roc_auc",
}

NUM_COLS_DEFAULT = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]
CAT_COLS_DEFAULT = [
    "Sex",
    "ChestPainType",
    "FastingBS",
    "RestingECG",
    "ExerciseAngina",
    "ST_Slope",
]
MULTI_CAT = ["ChestPainType", "RestingECG", "ST_Slope"]

INPUT_COLUMNS = [
    "Age",
    "Sex",
    "RestingBP",
    "Cholesterol",
    "FastingBS",
    "MaxHR",
    "ExerciseAngina",
    "Oldpeak",
    "ChestPainType_ASY",
    "ChestPainType_ATA",
    "ChestPainType_NAP",
    "ChestPainType_TA",
    "RestingECG_LVH",
    "RestingECG_Normal",
    "RestingECG_ST",
    "ST_Slope_Down",
    "ST_Slope_Flat",
    "ST_Slope_Up",
]
# ----------------------------
# Model hyperparameter configurations
# ----------------------------
CONFIG_RF = {
    "n_estimators": [200, 400, 800],
    "max_depth": [None, 6, 12],
    "class_weight": [None, "balanced"],
}
CONFIG_DT = {
    "criterion": ["gini", "entropy", "log_loss"],
    "max_depth": [None, 3, 5, 7, 9, 12],
    "min_samples_split": [2, 5, 10, 20],
    "min_samples_leaf": [1, 2, 4, 8],
    "max_features": [None, "sqrt", "log2"],
    "class_weight": [None, "balanced"],
    "ccp_alpha": [0.0, 0.001, 0.01],
}
CONFIG_LR = {"C": [0.01, 0.1, 1, 10], "penalty": ["l2"], "class_weight": [None, "balanced"]}

# ----------------------------
# Repository info
# ----------------------------
REPO_OWNER = "se4ai2526-uniba"
REPO_NAME = "CardioTrack"

# ----------------------------
# Great Expectations
# ----------------------------
SOURCE_NAME = "heart_data_source"
ASSET_NAME = "heart_failure"
SUITE_NAME = "heart_failure_data_quality"

# ----------------------------
# Paths
# ----------------------------
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RAW_DATA_DIR = DATA_DIR / "raw"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

RAW_PATH = RAW_DATA_DIR / "heart.csv"
PREPROCESSED_CSV = INTERIM_DATA_DIR / "preprocessed.csv"
TRAIN_CSV = PROCESSED_DATA_DIR / "train.csv"
TEST_CSV = PROCESSED_DATA_DIR / "test.csv"

MODELS_DIR = PROJ_ROOT / "models"
REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

METRICS_DIR = PROJ_ROOT / "metrics"
TEST_METRICS_DIR = METRICS_DIR / "test"

NOSEX_CSV = INTERIM_DATA_DIR / "preprocessed_no_sex_column.csv"
MALE_CSV = INTERIM_DATA_DIR / "preprocessed_male_only.csv"
FEMALE_CSV = INTERIM_DATA_DIR / "preprocessed_female_only.csv"

PREPROCESS_ARTIFACTS_DIR = INTERIM_DATA_DIR / "preprocess_artifacts"
SCALER_PATH = PREPROCESS_ARTIFACTS_DIR / "scaler.joblib"

MODEL_PATH = Path("models/nosex/random_forest.joblib")

CARD_PATHS = {
    "dataset card": DATA_DIR / "README.md",
    "model card": MODELS_DIR / "README.md",
}
