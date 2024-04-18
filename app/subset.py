import pandas as pd

# Load the CSV file
df = pd.read_csv('data-sampling/rba-dataset.csv')

# Calculate 10% of the total number of rows
subset_size = int(0.1 * len(df))

# Randomly select the subset
subset_df = df.sample(subset_size)

# Save the subset to a new CSV file
subset_df.to_csv('data-sampling/subset-rba-dataset.csv', index=False)