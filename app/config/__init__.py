from pydantic import Field
from functools import lru_cache
from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    """Configuration settings for the application"""
    APP_NAME: str = "rba-app"
    APP_ROOT_DIRECTORY: Path = Path.cwd()
    LOG_DIRECTORY: Path = APP_ROOT_DIRECTORY / "logs"
    
    KAGGLE_CONFIG_DIR: Path = Field(..., env="KAGGLE_CONFIG_DIR")
    DATASET: str = Field(default="dasgroup/rba-dataset", env="DATASET")
    DATA_DIRECTORY: str = "data"
    SAMPLE_DATA_DIRECTORY: Path = Path(DATA_DIRECTORY) / "sample_data"
    DATA_FILE: str = Field(default="rba-dataset.csv", env="DATA_FILE")
    SAMPLE_PERCENTAGE: float = Field(default=0.1, env="SAMPLE_PERCENT")

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
