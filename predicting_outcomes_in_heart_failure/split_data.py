import pandas as pd
from sklearn.model_selection import train_test_split
from loguru import logger
from config import (
    PREPROCESSED_CSV,
    INTERIM_DATA_DIR,
    TRAIN_CSV,
    TEST_CSV,
    TARGET_COL,
    RANDOM_STATE,
    TEST_SIZE
)


def split():
    logger.info("Starting data split process...")

    if not PREPROCESSED_CSV.exists():
        logger.error(f"Missing {PREPROCESSED_CSV}. Run preprocess.py first.")
        raise FileNotFoundError(f"Missing {PREPROCESSED_CSV}. Run preprocess.py first.")

    # Load dataset
    df = pd.read_csv(PREPROCESSED_CSV)
    logger.info(f"Loaded processed dataset: {PREPROCESSED_CSV} (rows={len(df)}, cols={df.shape[1]})")

    # Split features and target
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    stratify_y = y if y.nunique() > 1 else None
    if stratify_y is None:
        logger.warning("Target variable has only one unique value — skipping stratification.")
    else:
        logger.debug("Using stratified split based on target distribution.")

    # Perform split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=stratify_y,
        random_state=RANDOM_STATE,
        shuffle=True
    )
    logger.info(f"Performed train/test split with test_size={TEST_SIZE}")

    # Recombine for saving
    train_df = X_train.copy(); train_df[TARGET_COL] = y_train.values
    test_df  = X_test.copy();  test_df[TARGET_COL]  = y_test.values

    # Save to disk
    train_df.to_csv(TRAIN_CSV, index=False)
    test_df.to_csv(TEST_CSV, index=False)

    logger.success(f"Saved training set to {TRAIN_CSV} (rows={len(train_df)})")
    logger.success(f"Saved test set to {TEST_CSV} (rows={len(test_df)})")

    # Log class distribution
    train_counts = train_df[TARGET_COL].value_counts().to_dict()
    test_counts  = test_df[TARGET_COL].value_counts().to_dict()
    logger.info(f"Class distribution — TRAIN: {train_counts} | TEST: {test_counts}")

    logger.success("Data splitting completed successfully.")


if __name__ == "__main__":
    INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)
    split()
