from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path

from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import FeatureDrift
from loguru import logger
import pandas as pd
from predicting_outcomes_in_heart_failure.config import CAT_FEATURES


def _read_state(state_path: Path) -> dict:
    if not state_path.exists():
        return {"last_processed_rows": 0}
    try:
        return json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        return {"last_processed_rows": 0}


def _write_state(state_path: Path, state: dict) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

def _coerce_bool_like(df: pd.DataFrame) -> pd.DataFrame:
    """Converts columns with True/False values ​​to 0/1 int."""
    bool_map = {"true": 1, "false": 0, "1": 1, "0": 0, True: 1, False: 0}
    for col in df.columns:
        s = df[col]

        if s.dtype == "object":
            lowered = s.astype(str).str.strip().str.lower()
            uniq = set(lowered.dropna().unique().tolist())
            if uniq.issubset({"true", "false", "0", "1"}):
                df[col] = lowered.map(bool_map).astype("int64")
        elif s.dtype == "bool":
            df[col] = s.astype("int64")
    return df

def run_drift_if_enough_rows(
    *,
    reference_csv: Path,
    production_csv: Path,
    reports_dir: Path,
    min_rows: int = 20,
    feature_columns: list[str] | None = None,
    state_path: Path | None = None,
) -> dict:
    if not production_csv.exists():
        return {"ran": False, "reason": "production_csv_missing", "path": str(production_csv)}

    try:
        n_rows = sum(1 for _ in production_csv.open("r", encoding="utf-8")) - 1
    except Exception as e:
        logger.exception(f"Failed counting rows in {production_csv}: {e}")
        return {"ran": False, "reason": "count_failed", "error": str(e)}

    if n_rows < min_rows:
        return {"ran": False, "reason": "not_enough_rows", "n_rows": n_rows, "min_rows": min_rows}

    last_processed = 0
    if state_path is not None:
        state = _read_state(state_path)
        last_processed = int(state.get("last_processed_rows", 0))
        if n_rows == last_processed:
            return {"ran": False, "reason": "no_new_rows", "n_rows": n_rows}

    ref_df = pd.read_csv(reference_csv)
    prod_df = pd.read_csv(production_csv)
    
    ref_df = _coerce_bool_like(ref_df)
    prod_df = _coerce_bool_like(prod_df)


    if feature_columns is not None:
        ref_df = ref_df[feature_columns]
        prod_df = prod_df[feature_columns]

    ref_ds = Dataset(ref_df, label=None, cat_features=CAT_FEATURES)
    prod_ds = Dataset(prod_df, label=None, cat_features=CAT_FEATURES)

    check = FeatureDrift()

    try:
        result = check.run(train_dataset=ref_ds, test_dataset=prod_ds)
    except TypeError:
        result = check.run(ref_ds, prod_ds)

    reports_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y-%m-%d_%H-%M-%S")
    report_path = reports_dir / f"drift_result_{ts}.json"

    raw_json = result.to_json(with_display=False)

    report_path.write_text(raw_json, encoding="utf-8")

    logger.success(f"Deepchecks drift report generated: {report_path}")

    if state_path is not None:
        _write_state(state_path, {"last_processed_rows": n_rows, "last_report": str(report_path)})

    return {
        "ran": True,
        "n_rows": n_rows,
        "report_path": str(report_path),
        "passed": bool(getattr(result, "passed", True)),
    }
