from pydantic import Field
from functools import lru_cache
from pydantic_settings import BaseSettings
from pathlib import Path

class Config(BaseSettings):
    """Configuration settings for the application"""

    APP_ROOT_DIRECTORY: Path = Path.cwd()
    LOG_DIRECTORY: Path = APP_ROOT_DIRECTORY / "logs"
    
    KAGGLE_CONFIG_DIR: Path = Field(..., env="KAGGLE_CONFIG_DIR")
    DATASET: str = Field(..., env="DATASET")
    DATA_DIR: Path = Path("data")
    DATA_FILE: str = Field(..., env="DATA_FILE")
    SAMPLE_DATA_FILE: str = Field(..., env="SAMPLE_DATA_FILE")
    SUBSET_DATA_FILE: str = "Subset_" + DATA_FILE

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

@lru_cache()
def get_config() -> Config:
    """Get the configuration settings for the Minecraft Server Manager

    Returns:
        Config: The configuration settings
    """
    return Config()
