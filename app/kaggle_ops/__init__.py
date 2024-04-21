# kaggle_op/download.py
import os
import kaggle
from pathlib import Path
from config import get_settings

def get_kaggle_dataset():
    config = get_settings()

    # Ensure existing data directory
    data_directory = Path(config.DATA_DIRECTORY)
    data_directory.mkdir(parents=True, exist_ok=True)

    data_file_path = data_directory / config.DATA_FILE

    if not data_file_path.exists():
        # Path to kaggle.json
        print("Downloading dataset from Kaggle...")
        os.environ['KAGGLE_CONFIG_DIR'] = str(config.KAGGLE_CONFIG_DIR)
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(config.DATASET, path=config.DATA_DIRECTORY, unzip=True)
        print(f"Dataset downloaded at {data_file_path}")
    else:
        print(f"Dataset already downloaded at {data_file_path}")

    return data_file_path
