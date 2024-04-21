"""
============================
DS675 Yevgeniy Kim
Risk Based Authentication
============================
"""

from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import sklearn.model_selection
import datetime
import matplotlib.pyplot as plt
from sklearn import ensemble
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from pathlib import Path
from config import get_settings

config = get_settings()

# Ensure existing data directory
data_directory = Path(config.DATA_DIRECTORY)
data_directory.mkdir(parents=True, exist_ok=True)
sample_data_directory = Path(config.SAMPLE_DATA_DIRECTORY)
sample_data_directory.mkdir(parents=True, exist_ok=True)
data_file_path = data_directory / config.DATA_FILE

# function to fill missing values with the mode
def fill_with_mode(df):
    return df.fillna(df.mode().iloc[0])

# Load the subset data
subset_df = pd.read_csv(data_file_path)
print("Data after loading:\n")
print(subset_df.head())
def get_RiskFactor(x, y):
    if x == True and y == True:
        return 10
    elif x == True and y == False:
        return 1
    elif x == False and y == False:
        return 0.1
    else:
        return 999
    
subset_df['RiskFactor'] = np.vectorize(get_RiskFactor)(subset_df['Is Attack IP'],subset_df['Is Account Takeover'])


print(subset_df.columns)
print(subset_df.dtypes)

print('**************Summary for initial training dataframe**************')
print('**************numeric variables**************')
print(subset_df.describe().transpose())
print()
print('**************categorical variables**************')
print(subset_df.describe(include=['object']).transpose())
print()
print('**************boolean variables**************')
print(subset_df.describe(include=['bool']).transpose())
print()
print('**************Checking for NaN**************')
print(subset_df.isna().sum())
print()

print("subset_df before droping rows where RiskFactor == 999")
print("len subset_df =",len(subset_df))
print("Count of rows with RiskFactor 10 = ", len(subset_df[(subset_df['RiskFactor']==10)]))
print("Count of rows with RiskFactor 1 = ", len(subset_df[(subset_df['RiskFactor']==1)]))
print("Count of rows with RiskFactor 0.1 = ", len(subset_df[(subset_df['RiskFactor']==0.1)]))
print("Count of rows with RiskFactor 999 = ", len(subset_df[(subset_df['RiskFactor']==999)]))

subset_df.drop(subset_df[subset_df.RiskFactor == 999].index, inplace=True)
print("len subset_df =",len(subset_df))

df_cleaned = subset_df.dropna(axis=1)

print("subset_df after droping rows where RiskFactor == 999")
print("len subset_df =",len(df_cleaned))
print("Count of rows with RiskFactor 10 = ", len(df_cleaned[(df_cleaned['RiskFactor']==10)]))
print("Count of rows with RiskFactor 1 = ", len(df_cleaned[(df_cleaned['RiskFactor']==1)]))
print("Count of rows with RiskFactor 0.1 = ", len(df_cleaned[(df_cleaned['RiskFactor']==0.1)]))
print("Count of rows with RiskFactor 999 = ", len(df_cleaned[(df_cleaned['RiskFactor']==999)]))

#df_cleaned['index'].astype('int64')
df_cleaned.set_index(['index'], inplace=True)
df_cleaned.reset_index()

# splitting dataframe by row index
slice=round(len(df_cleaned)*0.2)
print("len df cleaned =",len(df_cleaned))
clean_data = df_cleaned.iloc[slice:,:] #this is a dataset for training the model
clean_test = df_cleaned.iloc[:slice,:] #this is an unseen dataset for final validation
print(clean_data.head())

print("check if clean_data and clean test have rows of each risk factor 10, 1.0 and 0.1")
print("len clean_data =",len(clean_data))
print("Count of rows with RiskFactor 10 = ", len(clean_data[(clean_data['RiskFactor']==10)]))
print("Count of rows with RiskFactor 1 = ", len(clean_data[(clean_data['RiskFactor']==1)]))
print("Count of rows with RiskFactor 0.1 = ", len(clean_data[(clean_data['RiskFactor']==0.1)]))
print("Count of rows with RiskFactor 999 = ", len(clean_data[(clean_data['RiskFactor']==999)]))
print("len clean_data =",len(clean_test))
print("Count of rows with RiskFactor 10 = ", len(clean_test[(clean_test['RiskFactor']==10)]))
print("Count of rows with RiskFactor 1 = ", len(clean_test[(clean_test['RiskFactor']==1)]))
print("Count of rows with RiskFactor 0.1 = ", len(clean_test[(clean_test['RiskFactor']==0.1)]))
print("Count of rows with RiskFactor 999 = ", len(clean_test[(clean_test['RiskFactor']==999)]))

## Remove unused variables
clean_data = clean_data.drop(columns=['Login Timestamp', 'IP Address', 'OS Name and Version', 'User Agent String', 'Browser Name and Version'])
clean_test = clean_test.drop(columns=['Login Timestamp', 'IP Address', 'OS Name and Version', 'User Agent String', 'Browser Name and Version'])


## Create dummies for categorical variables
clean_dataOlen = len(clean_data)
clean_testOlen = len(clean_test)
#clean_test = clean_test.append(clean_data)
clean_data1 = pd.concat([clean_data, clean_test], ignore_index=True)
#clean_data = clean_data.append(clean_test)
clean_test1 = pd.concat([clean_test, clean_data], ignore_index=True)

clean_data1 = pd.get_dummies(clean_data, columns=['Country'])
clean_test1 = pd.get_dummies(clean_test, columns=['Country'])

clean_test = clean_test1[:clean_testOlen]
clean_data = clean_data1[:clean_dataOlen]

print("clean data head")
print(clean_data.head())

X = clean_data.drop(columns=['RiskFactor'])
y = clean_data[['RiskFactor']]
print(type(y)) 

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print("\nFirst 5 rows of training data:\n", X_train[:5])
print("\nFirst 5 rows of test data:\n", X_test[:5])

print("Split the data into training and test sets for training the model")
print("number or rows X_train=",X_train.shape[0])
print("number or rows y_train=",y_train.shape[0])
print("number or rows X_test=",X_test.shape[0])
print("number or rows y_test=",y_test.shape[0])
print("number of columns X_train=",X_train.shape[1])
print("number of columns y_train=",y_train.shape[1])
print("number of columns  X_test=",X_test.shape[1])
print("number of columns  y_test=",y_test.shape[1])

print("X_test columns")
print(X_train.dtypes)
print("y_test columns")
print(y_train.dtypes)

## Random Forest model - training
params = {'n_estimators': 200, 'max_features': 10, 'n_jobs': -1,
          'random_state': 123}
rf_clf = ensemble.RandomForestRegressor(**params)
rf_clf.fit(X_train, y_train)
feature_importances = rf_clf.feature_importances_
print ()
print("******X_train, y_train fit******")
rf_score = rf_clf.score(X_train, y_train, sample_weight=None)
print ("mean accuracy = ", rf_score)
mse = mean_squared_error(y_test, rf_clf.predict(X_test))
print('RMSE =  ', mse**0.5)

## Plot importances
feature_importances = rf_clf.feature_importances_
indices = np.argsort(feature_importances)
df_feature_importance = pd.DataFrame(feature_importances, index = X_train.columns, \
                                     columns=['importance']).sort_values('importance',ascending=False)

plt.title('Feature Importances')
plt.barh(range(len(indices)), feature_importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [X_train.columns[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()

## Final model using all data for training 
features = clean_test.drop(columns=['RiskFactor'])
target = clean_test[['RiskFactor']]
rf_clf.fit(features,target)
feature_importances = rf_clf.feature_importances_
predicted=rf_clf.predict(features)
print ()
print("******features, target fit******")
rf_score = rf_clf.score(features,target, sample_weight=None)
print ("mean accuracy = ", rf_score)
mse = mean_squared_error(target, predicted)
print('RMSE =  ', mse**0.5)
print ()
print("******RMSE for test set with Random Forest Classifier******")
RoundPred = [round(value) for value in predicted]
mse = mean_squared_error(target, RoundPred)
print('RMSE =  ', mse**0.5)

## Plotting predictions vs observed
plt.figure(1)
plt.plot(target, predicted, 'bo')
plt.xlabel('Observed')
plt.ylabel('Predicted')
plt.show()
