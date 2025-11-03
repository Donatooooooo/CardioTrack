from loguru import logger
import typer
import kagglehub
import shutil
import os


from predicting_outcomes_in_heart_failure.config import RAW_DATA_DIR, DATASET_NAME

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
