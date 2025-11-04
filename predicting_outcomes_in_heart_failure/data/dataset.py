import os
import shutil

import kagglehub
from loguru import logger
from predicting_outcomes_in_heart_failure.config import DATASET_NAME, RAW_DATA_DIR
import typer

app = typer.Typer()


@app.command()
def main():
    logger.info("Downloading dataset from Kaggle...")
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    path = kagglehub.dataset_download(DATASET_NAME)
    shutil.copytree(path, RAW_DATA_DIR, dirs_exist_ok=True)
    logger.success("Dataset downloaded successfully to {RAW_DATA_DIR}.")



if __name__ == "__main__":
    app()
