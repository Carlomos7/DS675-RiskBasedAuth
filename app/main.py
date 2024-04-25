from config import get_settings
from log_config import get_logger
from kaggle_ops import get_kaggle_dataset
from data_preparation import DataPrep, StratifiedSampler
from logistic_regression import LogisticRegressionModel

def main():
    
    # Get Kaggle dataset
    dataset = get_kaggle_dataset()
    
    # Stratifed sampling
    stratify = StratifiedSampler()
    stratify_col = 'Is Account Takeover'
    features = ['Country', 'Device Type', 'Is Attack IP', 'Is Account Takeover', 'Login Timestamp']
    timestamp_col = 'Login Timestamp'
    stratified_sample = stratify.stratified_sampling_pipeline(dataset, stratify_col, features, timestamp_col)
    
    # Data preparation
    data_prep = DataPrep()
    features.remove('Login Timestamp')
    features.append('Hour')
    features.append('Day of Week')
    encode_cols = ['Country', 'Device Type', 'Hour', 'Day of Week']
    prepared_data = data_prep.preprocess_data(stratified_sample, features, encode_cols)
    
    # Logistic regression model
    log_reg = LogisticRegressionModel()
    log_reg.run(prepared_data)
    
if __name__ == '__main__':
    main()