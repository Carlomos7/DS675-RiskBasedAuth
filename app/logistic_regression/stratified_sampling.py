import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from pathlib import Path
from config import get_settings
from log_config import get_logger
from kaggle_ops import get_kaggle_dataset
import numpy as np

def stratified_sampling():
    # Initialize logging and configuration
    log = get_logger('stratified_sampling')
    config = get_settings()

    # Load the full dataset
    log.info("Loading full dataset...")
    df = pd.read_csv(get_kaggle_dataset())

    # Performing stratified sampling to reduce dataset size
    log.info("Performing stratified sampling...")
    _, sample_df = train_test_split(df, test_size=config.SAMPLE_PERCENTAGE, stratify=df['Is Account Takeover'], random_state=42)
    
    # Selecting relevant features for RBA
    log.info("Selecting relevant features for RBA...")
    features = ['Country', 'Device Type', 'Is Attack IP', 'Is Account Takeover', 'Login Timestamp']
    sample_df = sample_df[features]

    # Convert timestamp to datetime and extract useful temporal features
    sample_df['Login Timestamp'] = pd.to_datetime(sample_df['Login Timestamp'])
    sample_df['Hour'] = sample_df['Login Timestamp'].dt.hour
    sample_df['Day of Week'] = sample_df['Login Timestamp'].dt.dayofweek
    sample_df.drop('Login Timestamp', axis=1, inplace=True)

    def reduce_cardinality(df, column, threshold):
        """Reduce the cardinality of a column by grouping infrequent values into 'Other'

        Args:
            df (pd.DataFrame): The DataFrame containing the column
            column (str): The column to reduce cardinality
            threshold (int): The threshold below which values are grouped into 'Other'
        """
        counts = df[column].value_counts()
        other = counts[counts < threshold].index
        df.loc[df[column].isin(other), column] = 'Other'
    
    # Reduce cardinality of the 'Country' and 'Device Type' columns
    reduce_cardinality(sample_df, 'Country', 100)
    reduce_cardinality(sample_df, 'Device Type', 100)

    # Identify numerical and categorical columns
    numerical_cols = sample_df.select_dtypes(include=np.number).columns
    categorical_cols = sample_df.select_dtypes(include='object').columns

    # Fill missing values in numerical columns with the median
    for col in numerical_cols:
        sample_df[col].fillna(sample_df[col].median(), inplace=True)

    # Fill missing values in categorical columns with the mode
    for col in categorical_cols:
        sample_df[col].fillna(sample_df[col].mode()[0], inplace=True)
    
    # Save the preprocessed data
    sample_data_directory = Path(config.SAMPLE_DATA_DIRECTORY)
    stratified_sample_filename = sample_data_directory / f"stratified_subset_{config.DATA_FILE}"
    log.info(f"Saving stratified sample...")
    sample_df.to_csv(stratified_sample_filename, index=False)
    log.info("Stratified sample exported to {stratified_sample_filename}")
    

stratified_sampling()
