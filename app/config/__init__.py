from pydantic import Field
from functools import lru_cache
from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    """Configuration settings for the application"""
    
    APP_NAME: str = "rba-app"
    APP_ROOT_DIRECTORY: Path = Path.cwd()
    LOG_DIRECTORY: Path = APP_ROOT_DIRECTORY / "logs"
    
    # Kaggle Configuration
    KAGGLE_CONFIG_PATH: Path = Field(..., env="KAGGLE_CONFIG_PATH")
    KAGGLE_DATASET_NAME: str = Field(default="dasgroup/rba-dataset", env="KAGGLE_DATASET_NAME")
    
    # Data Directories
    DATA_ROOT_DIRECTORY: Path = Path("data")
    KAGGLE_DATASET_DIRECTORY: Path = DATA_ROOT_DIRECTORY / "kaggle_dataset"
    SAMPLE_DATA_DIRECTORY: Path = DATA_ROOT_DIRECTORY / "sample_data"
    PLOTS_DIRECTORY: Path = DATA_ROOT_DIRECTORY / "plots"
    
    # Dataset and Sample Configuration
    DATA_CSV_FILENAME: str = Field(default="rba-dataset.csv", env="DATA_CSV_FILENAME")
    SAMPLE_DATA_PERCENTAGE: float = Field(default=0.1, env="SAMPLE_DATA_PERCENTAGE")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

@lru_cache()
def get_settings() -> Settings:
    """Get the configuration settings for the application

    Returns:
        Settings: The configuration settings
    """
    return Settings()
