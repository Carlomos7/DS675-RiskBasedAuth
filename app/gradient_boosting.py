from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OrdinalEncoder
from pathlib import Path
from config import get_settings
from kaggle_ops import get_kaggle_dataset
from log_config import get_logger
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Get the configuration settings
config = get_settings()
log = get_logger("gradient_boosting.py")
sample_percentage = config.SAMPLE_DATA_PERCENTAGE

# Load the subset data
log.info("Loading the subset data...")
sample_data_directory = Path(config.SAMPLE_DATA_DIRECTORY)
subset_filename = sample_data_directory / f"stratified_subset_{config.DATA_CSV_FILENAME}"
df_subset = pd.read_csv(subset_filename)

# Separate the features and target variable
log.info("Separating the features and target variable...")
def get_RiskFactor(x, y):
    if x == True and y == True:
        return 10
    elif x == True and y == False:
        return 1
    elif x == False and y == False:
        return 0.1
    else:
        return 999

df_subset['RiskFactor'] = np.vectorize(get_RiskFactor)(df_subset['Is Attack IP'], df_subset['Is Account Takeover'])
target_variable = 'RiskFactor'
X = df_subset.drop('RiskFactor', axis=1)
y = df_subset['RiskFactor']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Identify categorical columns
categorical_cols = X_train.select_dtypes(include='object').columns

# One-hot encoding
log.info("Performing one-hot encoding...")
X_train = pd.get_dummies(X_train, columns=categorical_cols)
X_test = pd.get_dummies(X_test, columns=categorical_cols)

# Ensure the training and test sets have the same columns
X_train, X_test = X_train.align(X_test, join='left', axis=1)

# Fill missing values with 0 in test set (for categories not present in the test set)
X_test = X_test.fillna(0)


# Hyperparameter tuning
log.info("Performing hyperparameter tuning...")
param_grid = {
    'n_estimators': [50, 100 , 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3,4 ,5]
}

log.info("Fitting the model...")
gb = GradientBoostingRegressor(n_estimators=50, learning_rate=0.01, max_depth=3, random_state=42)
gb.fit(X_train, y_train)

""" 
gb = GradientBoostingRegressor(random_state=42)
grid_search = GridSearchCV(gb, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=2, n_jobs=-1)
log.info("Fitting the model...")
grid_search.fit(X_train, y_train) """

# Get the best model
""" log.info("Getting the best model...")
gb = grid_search.best_estimator_ """

# Make predictions
log.info("Making predictions...")
y_pred = gb.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
log.info("The mean squared error (MSE) on test set: {:.4f}".format(mse))

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

log.info(f"Mean Absolute Error (MAE): {mae}")
log.info(f"Mean Squared Error (MSE): {mse}")
log.info(f"Root Mean Squared Error (RMSE): {rmse}")
log.info(f"R-squared (R2 Score): {r2}")

# Calculate the absolute errors
errors = np.abs(y_test - y_pred)

# Create a scatter plot of actual vs predicted values, colored by error
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, c=errors, cmap='viridis', alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
plt.colorbar(label='Absolute Error')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted')
plt.show()