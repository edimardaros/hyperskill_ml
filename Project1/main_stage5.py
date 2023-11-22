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

# Select predictors and target variable
X = data[['rating', 'draft_round', 'bmi']]  # Selecting relevant predictors
y = data['salary']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Technique 1: Replace negative predictions with 0
predictions_zero = [max(0, pred) for pred in y_pred]

# Technique 2: Replace negative predictions with the median of the training part of y
median_y_train = y_train.median()
predictions_median = [max(median_y_train, pred) for pred in y_pred]

# Calculate MAPE for both options
mape_zero = mean_absolute_percentage_error(y_test, predictions_zero)
mape_median = mean_absolute_percentage_error(y_test, predictions_median)

# Print the best MAPE rounded to five decimal places
best_mape = round(min(mape_zero, mape_median), 5)
print(best_mape)