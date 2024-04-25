import pandas as pd
from pathlib import Path
from config import get_settings
from log_config import get_logger

class DataPrep:
    """Data preparation class for the RBA model"""
    
    def __init__(self):
        """Initialize the DataPrep class
        """
        self.log = get_logger(__name__)
        self.config = get_settings()
        self.sample_data_directory = Path(self.config.SAMPLE_DATA_DIRECTORY)

    def load_data(self, filename: str) -> pd.DataFrame:
        """Load the full dataset

        Args:
            filename (str): The filename of the dataset to load

        Returns:
            pd.DataFrame: The loaded dataset
        """
        self.log.info("Loading full dataset...")
        self.df = pd.read_csv(filename)
        self.log.info("Dataset loaded successfully.")
        return self.df
    
    def select_features(self, features: list) -> pd.DataFrame:
        """Select relevant features for RBA

        Args:
            features (list): The list of features to select

        Returns:
            pd.DataFrame: The DataFrame with selected features
        """
        
        self.log.info("Selecting relevant features for RBA...")
        self.df = self.df[features]
        self.log.info("Features selected successfully.")
        return self.df
    
    def encode_categorical_features(self, columns: list) -> pd.DataFrame:
        """Encode categorical features

        Args:
            columns (list): The list of columns to encode

        Returns:
            pd.DataFrame: The DataFrame with encoded categorical features
        """
        self.log.info("Encoding categorical features...")
        self.df = pd.get_dummies(self.df, columns=columns)
        self.log.info("Categorical features encoded successfully.")
        return self.df

    def export_data(self, filename: str) -> None:
        """Export preprocessed data to CSV
        
        Args:
            filename (str): The filename to export the preprocessed data
        """
        self.log.info("Exporting preprocessed data...")
        self.df.to_csv(filename, index=False)
        self.log.info(f"Preprocessed data exported to {filename}")
    
    def preprocess_data(self, filename: str, features: list, columns: list) -> str:
        """Preprocess data for the RBA model
        
        Args:
            filename (str): The filename of the dataset to load
            features (list): The list of features to select
            columns (list): The list of columns to encode
        """
        
        self.log.info("Preprocessing data...")
        self.load_data(filename)
        self.select_features(features)
        self.encode_categorical_features(columns)
        pre_processed_filename = self.sample_data_directory / 'pre-processed_subset_{self.config.DATA_FILE}'
        self.export_data(pre_processed_filename)
        
        return pre_processed_filename