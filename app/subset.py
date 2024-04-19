import os
import kaggle
import pandas as pd
import numpy as np
from pathlib import Path
from config import get_settings

config = get_settings()

# Ensure existing data directory
data_directory = Path(config.DATA_DIRECTORY)
data_directory.mkdir(parents=True, exist_ok=True)
sample_data_directory = Path(config.SAMPLE_DATA_DIRECTORY)
sample_data_directory.mkdir(parents=True, exist_ok=True)

data_file_path = data_directory / config.DATA_FILE

if not data_file_path.exists():
    # Path to kaggle.json
    print("Downloading dataset from Kaggle...")
    os.environ['KAGGLE_CONFIG_DIR'] = str(config.KAGGLE_CONFIG_DIR)
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(config.DATASET, path=config.DATA_DIRECTORY, unzip=True)
    print(f"Dataset downloaded at {data_file_path}")
else:
    print(f"Dataset already downloaded at {data_file_path}")

print("Creating dataframe...")
df = pd.read_csv(data_file_path)

cols = [
    'IP Address', 'Country', 'Region', 'City', 'ASN', 
    'User Agent String', 'OS Name and Version', 'Browser Name and Version', 
    'Device Type', 'Login Successful', 'Is Attack IP', 
    'Is Account Takeover', 'Round-Trip Time (RTT) [ms]'
]

# Filtering to only include necessary columns
print("Filtering relevant columns...")
df = df[cols]

# Stratified sampling based on the 'Country' column
stratify_col = 'Country'
print(f"Performing stratified sampling based on {stratify_col}...")

# Calculating subset size
subset_size = int(config.SAMPLE_PERCENTAGE * len(df))

# stratified sampling
subset_df = df.groupby(stratify_col, group_keys=False).apply(
    lambda x: x.sample(min(len(x), int(np.rint(subset_size * len(x) / len(df)))))).sample(frac=1).reset_index(drop=True)

# Save the sampled subset to a new CSV file
subset_filename = sample_data_directory / f"subset_{config.DATA_FILE}"
subset_df.to_csv(subset_filename, index=False)
print(f"Subset saved at {subset_filename}.")

"""
# Calculate 10% of the total number of rows
print("Calculating percentage of data to sample...")
subset_size = int(config.SAMPLE_PERCENTAGE * len(df))

# Randomly select the subset
print("Creating subset...")
subset_df = df.sample(subset_size)
"""
