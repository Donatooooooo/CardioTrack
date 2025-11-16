from unittest.mock import patch

from loguru import logger
import pandas as pd
from predicting_outcomes_in_heart_failure.data.split_data import (
    TARGET_COL,
    VARIANTS,
    _safe_train_test_split,
    split_one,
)
import pytest


# TEST: split_one()
@pytest.fixture
def csv_dir(tmp_path):
    """
    Create a temporary directory with mock CSV files for all variants.
    """

    paths = {}
    for variant, csv_name in VARIANTS.items():
        path = tmp_path / csv_name.name

        df = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4],
                "feature2": [10, 20, 30, 40],
                TARGET_COL: [0, 1, 0, 1],
            }
        )
        df.to_csv(path, index=False)
        paths[variant] = path
    return paths


@pytest.mark.parametrize("variant", ["all", "female", "male", "nosex"])
def test_split_one_creates_files(csv_dir, tmp_path, variant):
    """
    Test that split_one creates train/test CSV files correctly.
    """

    csv_path = csv_dir[variant]
    out_dir = tmp_path / "processed" / variant

    with patch(
        "predicting_outcomes_in_heart_failure.data.split_data.PROCESSED_DATA_DIR",
        tmp_path / "processed",
    ):
        split_one(csv_path, variant)

    train_file = out_dir / "train.csv"
    test_file = out_dir / "test.csv"

    assert train_file.exists()
    assert test_file.exists()

    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    assert TARGET_COL in train_df.columns
    assert TARGET_COL in test_df.columns

    assert len(train_df) + len(test_df) == 4
    assert set(train_df[TARGET_COL].unique()).issubset({0, 1})
    assert set(test_df[TARGET_COL].unique()).issubset({0, 1})


@pytest.mark.parametrize("variant", ["all", "female", "male", "nosex"])
def test_split_one_missing_csv_raises(tmp_path, variant):
    """
    Test that split_one outputs a warning log when CSV is missing.
    """

    from loguru import logger

    log_messages = []
    missing_csv = tmp_path / f"missing_{variant}.csv"
    handler_id = logger.add(log_messages.append, level="WARNING")

    try:
        split_one(missing_csv, variant)
    finally:
        logger.remove(handler_id)

    assert any("Missing CSV file" in msg for msg in log_messages)


class LogCapture:
    def __init__(self):
        self.messages = []

    def __call__(self, message):
        self.messages.append(message)


@pytest.mark.parametrize(
    "X, y, expected_classes, log_level, log_text",
    [
        (
            pd.DataFrame({"a": [1, 2, 3, 4]}),
            pd.Series([0, 1, 0, 1]),
            {0, 1},
            "DEBUG",
            "Stratified split executed successfully",
        ),
        (
            pd.DataFrame({"a": [1, 2, 3, 4]}),
            pd.Series([0, 0, 0, 0]),
            {0},
            "WARNING",
            "Target has only one class",
        ),
    ],
)
def test_safe_train_test_split_basic(X, y, expected_classes, log_level, log_text):
    log_capture = LogCapture()
    handler_id = logger.add(log_capture, level=log_level)

    try:
        X_tr, X_te, y_tr, _ = _safe_train_test_split(X, y, test_size=0.5, random_state=42)
    finally:
        logger.remove(handler_id)

    assert len(X_tr) + len(X_te) == len(X)
    assert set(y_tr).issubset(expected_classes)
    assert any(log_text in msg for msg in log_capture.messages)


def test_safe_train_test_split_fallback_exception():
    X = pd.DataFrame({"a": [1, 2, 3, 4]})
    y = pd.Series([0, 0, 1, 1])

    test_size = 0.75
    log_capture = []
    handler_id = logger.add(lambda msg: log_capture.append(msg), level="WARNING")

    try:
        X_tr, X_te, _, _ = _safe_train_test_split(X, y, test_size=test_size, random_state=42)
    finally:
        logger.remove(handler_id)

    assert len(X_tr) + len(X_te) == len(X)
    assert any("Falling back to non-stratified split" in msg for msg in log_capture)
