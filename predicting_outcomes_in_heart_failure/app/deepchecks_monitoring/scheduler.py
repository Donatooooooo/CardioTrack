from __future__ import annotations

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from loguru import logger
from predicting_outcomes_in_heart_failure.app.deepchecks_monitoring.drift_runner import (
    run_drift_if_enough_rows,
)
from predicting_outcomes_in_heart_failure.config import (
    INPUT_COLUMNS,
    PRODUCTION_CSV_PATH,
    REFERENCE_CSV,
    REPORTS_DIR,
    STATE_PATH,
)

scheduler = AsyncIOScheduler()


def drift_job() -> None:
    try:
        out = run_drift_if_enough_rows(
            reference_csv=REFERENCE_CSV,
            production_csv=PRODUCTION_CSV_PATH,
            reports_dir=REPORTS_DIR,
            min_rows=20,
            feature_columns=list(INPUT_COLUMNS),
            state_path=STATE_PATH,
        )
        logger.info(f"Drift job result: {out}")
    except Exception as e:
        logger.exception(f"Drift job failed: {e}")


def start_scheduler() -> None:
    if scheduler.running:
        return

    scheduler.add_job(
        drift_job,
        trigger=CronTrigger(minute="*/5"),
        id="deepchecks_drift_job",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
    )
    scheduler.start()
    logger.info("Deepchecks drift scheduler started (every 5 minutes).")


def shutdown_scheduler() -> None:
    if scheduler.running:
        scheduler.shutdown(wait=False)
        logger.info("Deepchecks drift scheduler stopped.")
