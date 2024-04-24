import os
import kaggle
from pathlib import Path
from config import get_settings
from log_config import get_logger
from getpass import getpass

log = get_logger(__name__)

def authenticate_kaggle():
    """Authenticate with Kaggle API using kaggle.json or environment variables."""
    kaggle_config_dir = Path.home() / ".kaggle"
    kaggle_config_file = kaggle_config_dir / "kaggle.json"
    
    if kaggle_config_file.exists():
        os.environ['KAGGLE_CONFIG_DIR'] = str(kaggle_config_dir)
    elif 'KAGGLE_USERNAME' in os.environ and 'KAGGLE_KEY' in os.environ:
        pass # Already authenticated via environment variables
    else:
        log.info("Kaggle API authentication required.")
        username = input("Enter your Kaggle username: ")
        password = getpass("Enter your Kaggle password: ")
        os.environ['KAGGLE_USERNAME'] = username
        os.environ['KAGGLE_KEY'] = password
    
    try:
        kaggle.api.authenticate()
        log.info("Kaggle API authenticated successfully.")
    except Exception as e:
        log.error(f"Failed to authenticate with Kaggle API: {e}")
        raise

def get_kaggle_dataset() -> Path:
    """Download the Kaggle dataset and return the path to the data file."""
    config = get_settings()
    data_directory = Path(config.DATA_DIRECTORY)
    data_directory.mkdir(parents=True, exist_ok=True)
    data_file_path = data_directory / config.DATA_FILE

    if not data_file_path.exists():
        log.info("Downloading dataset from Kaggle...")
        authenticate_kaggle()
        try:
            kaggle.api.dataset_download_files(config.DATASET, path=config.DATA_DIRECTORY, unzip=True)
            log.info(f"Dataset downloaded at {data_file_path}")
        except Exception as e:
            log.error(f"Failed to download dataset: {e}")
            raise
    else:
        log.info(f"Dataset already downloaded at {data_file_path}")

    return data_file_path
