from io import StringIO
from types import SimpleNamespace

import joblib
import pandas as pd
from predicting_outcomes_in_heart_failure.config import MODELS_DIR, TARGET_COL
from predicting_outcomes_in_heart_failure.modeling import evaluate
import pytest

TEST_PROCESSED_CSV = (
    "Age,Sex,RestingBP,Cholesterol,FastingBS,MaxHR,ExerciseAngina,Oldpeak,HeartDisease,"
    "ChestPainType_ASY,ChestPainType_ATA,ChestPainType_NAP,ChestPainType_TA,"
    "RestingECG_LVH,RestingECG_Normal,RestingECG_ST,ST_Slope_Down,ST_Slope_Flat,ST_Slope_Up\n"
    "-1.4322063372940435,1,0.41462668821399407,0.8574469341726604,0,1.3833394263306962,"
    "0,-0.8315022488659315,0,False,True,False,False,False,True,False,False,False,True\n"
    "-0.47805724933087407,0,1.5263596504719819,-1.183717051045972,0,0.7547357326016333,"
    "0,0.10625148648034725,1,False,False,True,False,False,True,False,False,True,False\n"
    "-1.7502560332817665,1,-0.14123979291499986,0.7450892836101669,0,-1.5239526571662194,"
    "0,-0.8315022488659315,0,False,True,False,False,False,False,True,False,False,True\n"
    "-0.5840738146601151,0,0.30345339198819526,-0.5470236978585087,0,-1.1310753485855551,"
    "1,0.5751283541534866,1,True,False,False,False,False,True,False,False,True,False\n"
    "0.05202557731533111,1,0.970493169342988,-0.9028229246397381,0,-0.5810471165726252,"
    "0,-0.8315022488659315,0,False,False,True,False,False,True,False,False,False,True\n"
    "-1.5382229026232843,1,-0.6971062740439938,1.7937606888601065,0,1.3047639646145632,"
    "0,-0.8315022488659315,0,False,False,True,False,False,True,False,False,False,True\n"
    "-0.9021235106478382,0,-0.14123979291499986,-0.11631937070228347,0,1.3047639646145632,"
    "0,-0.8315022488659315,0,False,True,False,False,False,True,False,False,False,True\n"
    "0.05202557731533111,1,-1.2529727551729877,-0.6593813484210022,0,0.2047075005887033,"
    "0,-0.8315022488659315,0,False,True,False,False,False,True,False,False,False,True\n"
    "-1.7502560332817665,1,0.41462668821399407,-0.6781076235147511,0,-0.2667452697080938,"
    "1,0.5751283541534866,1,True,False,False,False,False,True,False,False,True,False\n"
)


@pytest.fixture
def processed_df():
    df = pd.read_csv(StringIO(TEST_PROCESSED_CSV))
    return df


@pytest.fixture
def definition_X_test_and_y_test(processed_df):
    df = processed_df
    X_test = df.drop(columns=[TARGET_COL])
    y_test = df[TARGET_COL]

    return X_test, y_test


@pytest.fixture
def logreg_model():
    path = MODELS_DIR / "all" / "logreg.joblib"
    model = joblib.load(path)
    return model


@pytest.fixture
def decision_tree_model():
    path = MODELS_DIR / "all" / "decision_tree.joblib"
    return joblib.load(path)


@pytest.fixture
def random_forest_model():
    path = MODELS_DIR / "all" / "random_forest.joblib"
    return joblib.load(path)


@pytest.fixture
def sample_raw_df_single():
    """
    Single-row raw sample, similar to the one used in main().
    Used to test overall preprocessing shape / columns / NaNs.
    """
    return pd.DataFrame(
        {
            "Age": [54],
            "Sex": ["F"],
            "ChestPainType": ["ASY"],
            "RestingBP": [140],
            "Cholesterol": [239],
            "FastingBS": [0],
            "RestingECG": ["Normal"],
            "MaxHR": [160],
            "ExerciseAngina": ["N"],
            "Oldpeak": [0.0],
            "ST_Slope": ["Up"],
        }
    )


@pytest.fixture
def sample_raw_df_two_rows():
    """
    Two-row raw sample with variation in categorical features,
    used to test binary encodings and one-hot encoding.
    """
    return pd.DataFrame(
        {
            "Age": [50, 60],
            "Sex": ["M", "F"],
            "ChestPainType": ["ASY", "NAP"],
            "RestingBP": [130, 140],
            "Cholesterol": [220, 250],
            "FastingBS": [0, 1],
            "RestingECG": ["Normal", "ST"],
            "MaxHR": [150, 140],
            "ExerciseAngina": ["Y", "N"],
            "Oldpeak": [1.0, 0.0],
            "ST_Slope": ["Up", "Flat"],
        }
    )


@pytest.fixture
def sample_raw_df_only_asy_up():
    """
    Three-row raw sample where:
      - ChestPainType is always ASY
      - RestingECG is always Normal
      - ST_Slope is always Up

    Used to test that missing dummy columns (NAP, TA, ST, LVH, Flat, Down)
    are still present and filled with zeros.
    """
    return pd.DataFrame(
        {
            "Age": [50, 60, 55],
            "Sex": ["M", "F", "M"],
            "ChestPainType": ["ASY", "ASY", "ASY"],
            "RestingBP": [130, 140, 135],
            "Cholesterol": [220, 250, 230],
            "FastingBS": [0, 1, 0],
            "RestingECG": ["Normal", "Normal", "Normal"],
            "MaxHR": [150, 140, 145],
            "ExerciseAngina": ["Y", "N", "Y"],
            "Oldpeak": [1.0, 0.0, 0.5],
            "ST_Slope": ["Up", "Up", "Up"],
        }
    )


@pytest.fixture
def dummy_logger(monkeypatch):
    class DummyLogger:
        def __init__(self):
            self.warnings = []
            self.infos = []
            self.errors = []
            self.successes = []

        def warning(self, msg):
            self.warnings.append(msg)

        def info(self, msg):
            self.infos.append(msg)

        def error(self, msg):
            self.errors.append(msg)

        def success(self, msg):
            self.successes.append(msg)

    logger = DummyLogger()
    monkeypatch.setattr(evaluate, "logger", logger)
    return logger


@pytest.fixture
def mlflow_no_runs(monkeypatch):
    class DummyMlflow:
        called_search_runs = 0

        class data:
            @staticmethod
            def from_pandas(*args, **kwargs):
                pytest.fail("mlflow.data.from_pandas should not be called when there are no runs")

        class sklearn:
            @staticmethod
            def log_model(*args, **kwargs):
                pytest.fail("mlflow.sklearn.log_model should not be called when there are no runs")

        @staticmethod
        def get_experiment_by_name(name):
            # we SImulate an empty experiment
            return SimpleNamespace(experiment_id="exp-123")

        @staticmethod
        def search_runs(experiment_ids, filter_string, order_by, max_results):
            DummyMlflow.called_search_runs += 1
            # Empty DataFrame → runs.empty == True
            return pd.DataFrame()

        @staticmethod
        def start_run(run_id):
            pytest.fail("mlflow.start_run should not be called when there are no runs")

        @staticmethod
        def log_input(*args, **kwargs):
            pytest.fail("mlflow.log_input should not be called when there are no runs")

        @staticmethod
        def log_metrics(*args, **kwargs):
            pytest.fail("mlflow.log_metrics should not be called when there are no runs")

    monkeypatch.setattr(evaluate, "mlflow", DummyMlflow)
    return DummyMlflow


@pytest.fixture
def mlflow_experiment_missing(monkeypatch):
    class DummyMlflow:
        called_get_experiment = 0
        called_search_runs = 0

        @staticmethod
        def get_experiment_by_name(name):
            DummyMlflow.called_get_experiment += 1
            return None  # not founded experiment

        @staticmethod
        def search_runs(*args, **kwargs):
            DummyMlflow.called_search_runs += 1
            return pd.DataFrame()

    monkeypatch.setattr(evaluate, "mlflow", DummyMlflow)
    return DummyMlflow
