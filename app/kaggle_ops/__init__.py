import os
import kaggle
from pathlib import Path
from config import get_settings
from log_config import get_logger
from getpass import getpass

log = get_logger(__name__)

def authenticate_kaggle():
    """Authenticate with Kaggle API using kaggle.json or environment variables."""
    config = get_settings()
    kaggle_config_path = config.KAGGLE_CONFIG_PATH
    
    if kaggle_config_path.exists():
        os.environ['KAGGLE_CONFIG_DIR'] = str(kaggle_config_path)
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
    data_filename = config.DATA_CSV_FILENAME
    dataset_dir = Path(config.KAGGLE_DATASET_DIRECTORY)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    data_file_path = dataset_dir / data_filename

    if not data_file_path.exists():
        log.info("Downloading dataset from Kaggle...")
        authenticate_kaggle()
        try:
            kaggle.api.dataset_download_files(config.KAGGLE_DATASET_NAME, path=str(dataset_dir), unzip=True)
            log.info(f"Dataset downloaded at {data_file_path}")
        except Exception as e:
            log.error(f"Failed to download dataset: {e}")
            raise
    else:
        log.info(f"Dataset already downloaded at {data_file_path}")

    return data_file_path
