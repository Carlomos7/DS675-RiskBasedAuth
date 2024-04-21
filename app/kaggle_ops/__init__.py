# kaggle_op/download.py
import os
import kaggle
from pathlib import Path
from config import get_settings
from log_config import get_logger

log = get_logger(__name__)

def get_kaggle_dataset() -> Path:
    """Download the Kaggle dataset and return the path to the data file

    Returns:
        Path: The path to the downloaded data file
    """
    config = get_settings()

    # Ensure existing data directory
    data_directory = Path(config.DATA_DIRECTORY)
    data_directory.mkdir(parents=True, exist_ok=True)

    data_file_path = data_directory / config.DATA_FILE

    if not data_file_path.exists():
        # Path to kaggle.json
        log.info("Downloading dataset from Kaggle...")
        os.environ['KAGGLE_CONFIG_DIR'] = str(config.KAGGLE_CONFIG_DIR)
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(config.DATASET, path=config.DATA_DIRECTORY, unzip=True)
        log.info(f"Dataset downloaded at {data_file_path}")
    else:
        log.info(f"Dataset already downloaded at {data_file_path}")

    return data_file_path
