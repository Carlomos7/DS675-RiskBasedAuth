from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import pandas as pd

# function to fill missing values with the mode
def fill_with_mode(df):
    return df.fillna(df.mode().iloc[0])

# Load the subset data
subset_df = pd.read_csv('data-sampling/subset-rba-dataset.csv')
print("Data after loading:\n", subset_df.head())

# Check for missing values
print("\nMissing values in each column:\n", subset_df.isnull().sum())

# Define preprocessing for numeric columns (normalize them)
numeric_features = subset_df.select_dtypes(include=['int64', 'float64']).columns
numeric_transformer = Pipeline(steps=[
    ('imputer', FunctionTransformer(fill_with_mode, validate=False)),
    ('scaler', StandardScaler())])

# Define preprocessing for categorical features (one-hot encode them)
categorical_features = subset_df.select_dtypes(include=['object']).columns
categorical_transformer = Pipeline(steps=[
    ('imputer', FunctionTransformer(fill_with_mode, validate=False)),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])


# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# Apply transformations to our data
X = preprocessor.fit_transform(subset_df)
print("\nData after preprocessing:\n", X[:5])

y = subset_df[['Is Attack IP', 'Is Account Takeover']]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print("\nFirst 5 rows of training data:\n", X_train[:5])
print("\nFirst 5 rows of test data:\n", X_test[:5])
