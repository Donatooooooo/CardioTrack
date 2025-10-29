from pathlib import Path
 
from dotenv import load_dotenv
from loguru import logger
 
# Load environment variables from .env file if it exists
load_dotenv()

EXPERIMENT_NAME = "Heart_Failure_Prediction"
DATASET_NAME = "fedesoriano/heart-failure-prediction"

# Paths
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
TEST_CSV  = PROCESSED_DATA_DIR / "test.csv"
 
MODELS_DIR  = PROJ_ROOT / "models"
REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
 
 
NUM_COLS_DEFAULT = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]
CAT_COLS_DEFAULT = ["Sex", "ChestPainType", "FastingBS", "RestingECG", "ExerciseAngina", "ST_Slope"]
 
TARGET_COL   = "HeartDisease"
RANDOM_STATE = 42
N_SPLITS = 5
TEST_SIZE = 0.30
 
SCORING = {
    "accuracy": "accuracy",
    "f1": "f1",
    "recall": "recall",
    "roc_auc": "roc_auc",
}
 
# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm
 
    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass