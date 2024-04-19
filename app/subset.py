import os
import kaggle
import pandas as pd
from pathlib import Path
from config import get_settings

config = get_settings()

# Ensure existing data directory
data_directory = Path(config.DATA_DIRECTORY)
data_directory.mkdir(parents=True, exist_ok=True)

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

# Calculate 10% of the total number of rows
print("Calculating percentage of data to sample...")
subset_size = int(config.SAMPLE_PERCENTAGE * len(df))

# Randomly select the subset
print("Creating subset...")
subset_df = df.sample(subset_size)

# Save the subset to a new CSV file
subset_df.to_csv(config.SAMPLE_DATA_DIRECTORY / f"subset_{config.DATA_FILE}", index=False)
print(f"Subset saved at {config.SAMPLE_DATA_DIRECTORY / f'subset_{config.DATA_FILE}'}.")
