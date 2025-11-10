import argparse
from pathlib import Path

from loguru import logger
import pandas as pd
from predicting_outcomes_in_heart_failure.config import (
    FEMALE_CSV,
    MALE_CSV,
    NOSEX_CSV,
    PREPROCESSED_CSV,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    TARGET_COL,
    TEST_SIZE,
)
from sklearn.model_selection import train_test_split

VARIANTS = {
    "all": PREPROCESSED_CSV,
    "female": FEMALE_CSV,
    "male": MALE_CSV,
    "nosex": NOSEX_CSV,
}

def _safe_train_test_split(X, y, test_size, random_state):
    """Perform a stratified train/test split with fallback if not possible."""
    stratify_y = y if y.nunique() > 1 else None
    try:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y,
            test_size=test_size,
            stratify=stratify_y,
            random_state=random_state,
            shuffle=True
        )
        if stratify_y is None:
            logger.warning("Target has only one class — performing non-stratified split.")
        else:
            logger.debug("Stratified split executed successfully.")
        return X_tr, X_te, y_tr, y_te
    except ValueError as e:
        logger.warning(
            f"Stratified split failed ({e}). Falling back to non-stratified split."
        )
        return train_test_split(
            X, y,
            test_size=test_size,
            stratify=None,
            random_state=random_state,
            shuffle=True
        )
        
def split_one(csv_path: Path, variant: str):
    """Split a specific variant (all/female/male/nosex) into train/test sets."""
    if not csv_path.exists():
        logger.warning(f"[{variant}] Missing CSV file: {csv_path} — skipping.")
        return

    df = pd.read_csv(csv_path)
    logger.info(f"[{variant}] Loaded {csv_path} (rows={len(df)}, cols={df.shape[1]})")

    if TARGET_COL not in df.columns:
        raise ValueError(f"[{variant}] Target column '{TARGET_COL}' not found in {csv_path}")

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].astype(int)

    X_train, X_test, y_train, y_test = _safe_train_test_split(
        X, y, TEST_SIZE, RANDOM_STATE
    )

    train_df = X_train.copy()
    train_df[TARGET_COL] = y_train.values
    test_df  = X_test.copy()
    test_df[TARGET_COL]  = y_test.values

    out_dir = PROCESSED_DATA_DIR / variant
    out_dir.mkdir(parents=True, exist_ok=True)
    train_p = out_dir / "train.csv"
    test_p  = out_dir / "test.csv"

    train_df.to_csv(train_p, index=False)
    test_df.to_csv(test_p, index=False)

    logger.success(f"[{variant}] Saved TRAIN -> {train_p} (rows={len(train_df)})")
    logger.success(f"[{variant}] Saved TEST  -> {test_p} (rows={len(test_df)})")

    train_counts = train_df[TARGET_COL].value_counts().to_dict()
    test_counts  = test_df[TARGET_COL].value_counts().to_dict()
    logger.info(f"[{variant}] Class distribution — TRAIN: {train_counts} | TEST: {test_counts}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        type=str,
        choices=list(VARIANTS.keys()),
        required=True,
        help="Data variant to split: all, female, male, or nosex.",
    )
    args = parser.parse_args()

    variant = args.variant
    csv_path = VARIANTS[variant]

    logger.info(f"Starting splitting for variant='{variant}'")
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    split_one(csv_path, variant)
    logger.success(f"Splitting completed for variant='{variant}'")

if __name__ == "__main__":
    main()