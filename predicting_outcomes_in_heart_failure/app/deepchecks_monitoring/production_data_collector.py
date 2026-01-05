from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from loguru import logger
import pandas as pd


def append_predictions_to_csv(
    *,
    csv_path: Path,
    endpoint: str,
    X: pd.DataFrame,
    y_pred: list[int] | int,
    feature_columns: list[str] | None = None,
) -> None:
    """
    Appende su CSV i dati di produzione (input + prediction).
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Normalize predictions
    y_list = [y_pred] if isinstance(y_pred, int) else y_pred

    df = X.copy()

    if feature_columns is not None:
        missing = [c for c in feature_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Missing expected feature columns in production batch: {missing}")
        df = df[feature_columns]

    if len(df) != len(y_list):
        raise ValueError(
            f"Row mismatch: X has {len(df)} rows "
            f"but predictions has {len(y_list)} items"
        )
    df.insert(0, "timestamp_utc", datetime.now(UTC).isoformat())
    df.insert(1, "endpoint", endpoint)
    df["prediction"] = y_list

    write_header = not csv_path.exists()
    df.to_csv(csv_path, mode="a", index=False, header=write_header)
    logger.info(f"Appended {len(df)} rows to production CSV: {csv_path}")
