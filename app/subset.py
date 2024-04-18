import os
import kaggle
import pandas as pd

# path to kaggle.json
os.environ['KAGGLE_CONFIG_DIR'] = "~/.kaggle/kaggle.json"
kaggle.api.authenticate()
dataset_name = 'dasgroup/rba-dataset'
download_path = '../data-sampling/'
kaggle.api.dataset_download_files(dataset_name, path=download_path, unzip=True)

file_name = 'rba-dataset.csv'
df = pd.read_csv(os.path.join(download_path, file_name))

# Calculate 10% of the total number of rows
subset_size = int(0.1 * len(df))

# Randomly select the subset
subset_df = df.sample(subset_size)

# Save the subset to a new CSV file
subset_df.to_csv('data-sampling/subset-rba-dataset.csv', index=False)