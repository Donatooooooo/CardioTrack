import joblib
from loguru import logger
import pandas as pd
from predicting_outcomes_in_heart_failure.config import (
    FEMALE_CSV,
    INTERIM_DATA_DIR,
    MALE_CSV,
    NOSEX_CSV,
    NUM_COLS_DEFAULT,
    PREPROCESS_ARTIFACTS_DIR,
    PREPROCESSED_CSV,
    RAW_PATH,
    SCALER_PATH,
    TARGET_COL,
)
from sklearn.preprocessing import StandardScaler


def save_scaler_artifact(scaler: StandardScaler):
    """Save only the fitted scaler used during preprocessing for inference."""
    PREPROCESS_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)
    logger.success(f"Saved StandardScaler to {SCALER_PATH}")


def generate_gender_splits(df: pd.DataFrame):
    """Create and save gender-based CSV splits (female, male, nosex)."""
    if "Sex" in df.columns:
        df_female = df[df["Sex"] == 0].copy()
        df_female.to_csv(FEMALE_CSV, index=False)
        logger.success(f"Saved female-only dataset: {FEMALE_CSV} (rows={len(df_female)})")

    if "Sex" in df.columns:
        df_male = df[df["Sex"] == 1].copy()
        df_male.to_csv(MALE_CSV, index=False)
        logger.success(f"Saved male-only dataset: {MALE_CSV} (rows={len(df_male)})")

    df_nosex = df.drop(columns=["Sex"], errors="ignore").copy()
    df_nosex.to_csv(NOSEX_CSV, index=False)
    logger.success(f"Saved dataset without 'Sex': {NOSEX_CSV} (rows={len(df_nosex)})")


def preprocessing():
    """Run the full preprocessing pipeline on the raw heart dataset."""
    logger.info("Starting preprocessing pipeline...")

    if not RAW_PATH.exists():
        logger.error(f"Missing {RAW_PATH}. Put heart.csv under data/raw/ or adjust RAW_PATH.")
        raise FileNotFoundError(f"Missing {RAW_PATH}.")

    df = pd.read_csv(RAW_PATH)
    logger.info(f"Loaded dataset: {RAW_PATH} (rows={len(df)}, cols={df.shape[1]})")

    if len(df) < 2:
        raise ValueError("Preprocessing requires at least 2 rows, got only 1.")

    # Ensure target is integer
    df[TARGET_COL] = df[TARGET_COL].astype(int)

    # Remove invalid RestingBP rows
    if "RestingBP" in df.columns:
        before = len(df)
        df = df[df["RestingBP"] != 0].reset_index(drop=True)
        removed = before - len(df)
        if removed > 0:
            logger.warning(f"Removed {removed} rows with RestingBP == 0")

    # Impute missing/zero Cholesterol
    if "Cholesterol" in df.columns:
        zero_mask = df["Cholesterol"] == 0
        if zero_mask.any():
            median_chol = df.loc[~zero_mask, "Cholesterol"].median()
            df.loc[zero_mask, "Cholesterol"] = median_chol
            logger.info(f"Imputed {zero_mask.sum()} Cholesterol==0 with median={median_chol}")

    # Encode binary categorical features
    if "Sex" in df.columns:
        df["Sex"] = df["Sex"].map({"M": 1, "F": 0}).astype(int)
        logger.debug("Encoded 'Sex' as binary.")

    if "ExerciseAngina" in df.columns:
        df["ExerciseAngina"] = df["ExerciseAngina"].map({"Y": 1, "N": 0}).astype(int)
        logger.debug("Encoded 'ExerciseAngina' as binary.")

    # One-hot encode multi-category features
    multi_cat = [c for c in ["ChestPainType", "RestingECG", "ST_Slope"] if c in df.columns]
    df = pd.get_dummies(df, columns=multi_cat, drop_first=False)
    logger.debug(f"One-hot encoded columns: {multi_cat}")

    # Scale numerical columns
    num_cols = [c for c in NUM_COLS_DEFAULT if c in df.columns and c != TARGET_COL]
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    logger.info(f"Scaled numerical features: {num_cols}")

    # Save processed dataset
    df.to_csv(PREPROCESSED_CSV, index=False)
    logger.success(
        "Saved preprocessed dataset: %s (rows=%d, cols=%d)", PREPROCESSED_CSV, len(df), df.shape[1]
    )

    # Log class distribution
    count_0 = (df[TARGET_COL] == 0).sum()
    count_1 = (df[TARGET_COL] == 1).sum()
    logger.info(f"Target balance — 0: {count_0} | 1: {count_1}")

    save_scaler_artifact(scaler)

    logger.success("Preprocessing completed successfully.")
    return df


if __name__ == "__main__":
    INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df_processed = preprocessing()
    generate_gender_splits(df_processed)
