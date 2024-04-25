import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
from config import get_settings
from log_config import get_logger

def preprocess_data():
    # Initialize logging and configuration
    log = get_logger(__name__)
    config = get_settings() 
    sample_data_directory = Path(config.SAMPLE_DATA_DIRECTORY)
    stratified_sample_filename = sample_data_directory / f"stratified_subset_{config.DATA_FILE}"
    sample_df = pd.read_csv(stratified_sample_filename)
    
    # Selecting relevant features for RBA
    log.info("Selecting relevant features for RBA...")
    features = ['Country', 'Device Type', 'Is Attack IP', 'Is Account Takeover', 'Hour', 'Day of Week']
    sample_df = sample_df[features]

    # Encode categorical features
    log.info("Encoding categorical features...")
    sample_df = pd.get_dummies(sample_df, columns=['Country', 'Device Type', 'Hour', 'Day of Week'])

    # Write modified DataFrame back to CSV
    log.info("Exporting preprocessed data...")
    preprocessed_data = sample_data_directory / f"pre-processed_subset_{config.DATA_FILE}"
    sample_df.to_csv(preprocessed_data, index=False)
    log.info(f"Preprocessed data exported to {preprocessed_data}")
    
preprocess_data()