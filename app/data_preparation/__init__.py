"""
This module is responsible for preparing the data for the logistic regression model.
"""

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from config import get_settings
from log_config import get_logger

class BasePreprocessor:
    """Base class for data preprocessing operations"""
    def __init__(self):
        """Initialize the BasePreprocessor class
        """
        self.log = get_logger(__name__)
        self.config = get_settings()
        self.kaggle_csv = self.config.DATA_CSV_FILENAME
        self.sample_data_directory = Path(self.config.SAMPLE_DATA_DIRECTORY)

    def load_data(self, filename: str) -> pd.DataFrame:
        """Load the stratified sample

        Args:
            filename (str): The filename of the stratified sample

        Returns:
            pd.DataFrame: The loaded stratified sample
        """
        self.log.info(f"Loading data from {filename}...")
        df = pd.read_csv(filename)
        self.log.info("Data loaded successfully.")
        return df

    def select_features(self, df: pd.DataFrame, features: list) -> pd.DataFrame:
        """Select relevant features for RBA

        Args:
            df (pd.DataFrame): The DataFrame to select features
            features (list): The list of features to select

        Returns:
            pd.DataFrame: The DataFrame with selected features
        """
        self.log.info("Selecting relevant features...")
        df = df[features]
        self.log.info("Features selected successfully.")
        return df

    def export_data(self, df: pd.DataFrame, filename: str) -> None:
        """Export preprocessed data to CSV

        Args:
            df (pd.DataFrame): The DataFrame to export
            filename (str): The filename to export the preprocessed data
        """
        self.log.info("Exporting preprocessed data...")
        df.to_csv(filename, index=False)
        self.log.info(f"Preprocessed data exported to {filename}")

class StratifiedSampler(BasePreprocessor):
    """Stratified sampling class for the RBA model"""
    
    def __init__(self):
        """Initialize the StratifiedSampler class
        """
        super().__init__()
        self.sample_percent = self.config.SAMPLE_DATA_PERCENTAGE

    def perform_stratifed_sampling(self, df: pd.DataFrame, target: str) -> pd.DataFrame:
        """Perform stratified sampling on the dataset

        Args:
            df (pd.DataFrame): The DataFrame to perform stratified sampling
            target (str): The target column for stratified sampling

        Returns:
            pd.DataFrame: The stratified sample
        """
        _, sample_df = train_test_split(df, test_size=self.sample_percent, stratify=df[target], random_state=42)
        return sample_df

    def convert_timestamp(self, df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
        """Convert timestamp to datetime and extract useful temporal features

        Args:
            df (pd.DataFrame): The DataFrame to convert timestamp
            timestamp_col (str): The column containing the timestamp

        Returns:
            pd.DataFrame: The DataFrame with converted timestamp
        """
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        df['Hour'] = df[timestamp_col].dt.hour
        df['Day of Week'] = df[timestamp_col].dt.dayofweek
        df.drop(timestamp_col, axis=1, inplace=True)
        return df

    def reduce_cardinality(self, df: pd.DataFrame, column: str, threshold: int) -> pd.DataFrame:
        """Reduce the cardinality of a column by grouping infrequent values into 'Other'

        Args:
            df (pd.DataFrame): The DataFrame containing the column
            column (str): The column to reduce cardinality
            threshold (int): The threshold below which values are grouped into 'Other'

        Returns:
            pd.DataFrame: The DataFrame with reduced cardinality
        """
        counts = df[column].value_counts()
        other = counts[counts < threshold].index
        df.loc[df[column].isin(other), column] = 'Other'
        return df

    def fill_cat_mode(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """Fill missing values in categorical columns with the mode

        Args:
            df (pd.DataFrame): The DataFrame to fill missing values
            columns (list): The list of columns to fill

        Returns:
            pd.DataFrame: The DataFrame with filled missing values
        """
        for col in columns:
            df.fillna({col: df[col].mode()[0]}, inplace=True)
        return df

    def fill_num_median(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """Fill missing values in numerical columns with the median

        Args:
            df (pd.DataFrame): The DataFrame to fill missing values
            columns (list): The list of columns to fill

        Returns:
            pd.DataFrame: The DataFrame with filled missing values
        """
        for col in columns:
            df.fillna({col: df[col].median()}, inplace=True)
        return df

    def stratified_sampling_pipeline(self, filename: str, target: str, features: list, timestamp_col: str) -> str:
        """Run the stratified sampling pipeline

        Args:
            filename (str): The filename of the dataset to load
            target (str): The target column for stratified sampling
            features (list): The list of features to select
            timestamp_col (str): The column containing the timestamp
        
        Returns:
            str: The filename of the stratified sample
        """
        df = self.load_data(filename)
        sample_df = self.perform_stratifed_sampling(df, target)
        sample_df = self.select_features(sample_df, features)
        sample_df = self.convert_timestamp(sample_df, timestamp_col)
        num_cols = sample_df.select_dtypes(include='number').columns
        cat_cols = sample_df.select_dtypes(include='object').columns
        for col in cat_cols:
            sample_df = self.reduce_cardinality(sample_df, col, 100)
        sample_df = self.fill_num_median(sample_df, num_cols)
        sample_df = self.fill_cat_mode(sample_df, cat_cols)
        stratified_sample_filename = self.sample_data_directory / f"stratified_subset_{self.kaggle_csv}"
        self.export_data(sample_df, stratified_sample_filename)
        return stratified_sample_filename

class DataPrep(BasePreprocessor):
    """Data preparation class for the RBA model"""
    def encode_categorical_features(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        self.log.info("Encoding categorical features...")
        df = pd.get_dummies(df, columns=columns)
        self.log.info("Categorical features encoded successfully.")
        return df

    def preprocess_data(self, filename: str, features: list, columns: list) -> str:
        """Preprocess data for the RBA model
        
        Args:
            filename (str): The filename of the dataset to load
            features (list): The list of features to select
            columns (list): The list of columns to encode
        """
        df = self.load_data(filename)
        df = self.select_features(df, features)
        df = self.encode_categorical_features(df, columns)
        pre_processed_filename = self.sample_data_directory / f'pre-processed_subset_{self.kaggle_csv}'
        self.export_data(df, pre_processed_filename)
        return pre_processed_filename
