import os
import kaggle
import pandas as pd
import numpy as np
from config import get_settings
from pathlib import Path
from kaggle_ops import get_kaggle_dataset
from log_config import get_logger

config = get_settings()
log = get_logger("subset.py")

data_file_path = get_kaggle_dataset()
sample_data_directory = Path(config.SAMPLE_DATA_DIRECTORY)
sample_data_directory.mkdir(parents=True, exist_ok=True)

log.info("Creating dataframe...")
df = pd.read_csv(data_file_path)

cols = [
    'IP Address', 'Country', 'Region', 'City', 'ASN', 
    'User Agent String', 'OS Name and Version', 'Browser Name and Version', 
    'Device Type', 'Login Successful', 'Is Attack IP', 
    'Is Account Takeover'
]

# Filtering to only include necessary columns
log.debug("Filtering relevant columns...")
df = df[cols]

# Stratified sampling based on the 'Country' column
stratify_col = 'Country'
log.debug(f"Performing stratified sampling based on {stratify_col}...")

# Calculating subset size
subset_size = int(config.SAMPLE_PERCENTAGE * len(df))

# Simplified stratified sampling using groupby and sample, excluding group keys.
subset_df = df.groupby(stratify_col, group_keys=False).apply(
    lambda x: x.sample(min(len(x), subset_size)), include_groups=False).sample(frac=1).reset_index(drop=True)
# Referenced: https://stackoverflow.com/questions/44114463/stratified-sampling-in-pandas

# Save the sampled subset to a new CSV file
subset_filename = sample_data_directory / f"subset_{config.DATA_FILE}"
subset_df.to_csv(subset_filename, index=False)
log.info(f"Subset saved at {subset_filename}.")

"""
# Calculate 10% of the total number of rows
print("Calculating percentage of data to sample...")
subset_size = int(config.SAMPLE_PERCENTAGE * len(df))

# Randomly select the subset
print("Creating subset...")
subset_df = df.sample(subset_size)
"""
