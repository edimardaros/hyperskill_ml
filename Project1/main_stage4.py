import os
import requests

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error

# checking ../Data directory presence
if not os.path.exists('./Data'):
    os.mkdir('./Data')

# download data if it is unavailable
if 'data.csv' not in os.listdir('./Data'):
    url = "https://www.dropbox.com/s/3cml50uv7zm46ly/data.csv?dl=1"
    r = requests.get(url, allow_redirects=True)
    open('./Data/data.csv', 'wb').write(r.content)

# Step 1
# read data
data = pd.read_csv('./Data/data.csv')


# Calculating the correlation matrix
correlation_matrix = data.corr()

# Identifying variables where the correlation coefficient is greater than 0.2
high_corr_variables = correlation_matrix.abs().unstack().sort_values(kind="quicksort", ascending=False)
high_corr_variables = high_corr_variables[high_corr_variables > 0.2]
high_corr_variables = high_corr_variables[high_corr_variables < 1] # exclude self-correlation

# Selecting pairs of highly correlated variables
high_corr_pairs = high_corr_variables.reset_index()
high_corr_pairs.columns = ['Variable1', 'Variable2', 'Correlation']
high_corr_pairs = high_corr_pairs.drop_duplicates(subset='Correlation')

# Extracting the unique variables from these pairs
high_corr_vars_unique = np.unique(high_corr_pairs[['Variable1', 'Variable2']])

# Defining X (predictors) and y (target)
X = data.drop('salary', axis=1)
y = data['salary']

# Splitting the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)

# Subsets of predictors
subsets = {
    'Without age': X_train.drop('age', axis=1),
    'Without experience': X_train.drop('experience', axis=1),
    'Without rating': X_train.drop('rating', axis=1),
    'Without age and experience': X_train.drop(['age', 'experience'], axis=1),
    'Without experience and rating': X_train.drop(['experience', 'rating'], axis=1),
    'Without age and rating': X_train.drop(['age', 'rating'], axis=1),
}

# Fit models and calculate MAPE for each subset
mape_scores = {}

for subset_name, subset_X in subsets.items():
    model = LinearRegression().fit(subset_X, y_train)
    X_test_subset = X_test[subset_X.columns]
    y_pred = model.predict(X_test_subset)
    mape_score = mean_absolute_percentage_error(y_test, y_pred)
    mape_scores[subset_name] = mape_score

best_subset = min(mape_scores, key=mape_scores.get)
lowest_mape = mape_scores[best_subset]

# print(f'Lowest MAPE: {lowest_mape}, Best Model: {best_subset}')
print(lowest_mape)