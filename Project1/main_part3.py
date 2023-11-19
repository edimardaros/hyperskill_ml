import os
import requests

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error as mape

# checking ../Data directory presence
if not os.path.exists('./Data'):
    os.mkdir('./Data')

# download data if it is unavailable
if 'data.csv' not in os.listdir('./Data'):
    url = "https://www.dropbox.com/s/3cml50uv7zm46ly/data.csv?dl=1"
    r = requests.get(url, allow_redirects=True)
    open('./Data/data.csv', 'wb').write(r.content)

# read data
data = pd.read_csv('./Data/data.csv')

# Select all columns for features except 'salary' for X, and 'salary' for y
X = data.drop('salary', axis=1)
y = data['salary']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)

# Fit the model
model = LinearRegression()
model.fit(X_train, y_train)

# Get the model coefficients
coefficients = model.coef_
intercept = model.intercept_

# Coefficients and intercept
(coefficients, intercept)

# print(coefficients)
# print(intercept)

# Adjusting the code to print the coefficients and intercept in the required format
# Coefficients separated by a comma
coefficients_str = ', '.join(f'{coef:.5e}' for coef in coefficients)
intercept_str = f'{intercept:.5e}'

print(coefficients_str)

# Display coefficients and intercept as a single comma-separated string
# model_parameters = f'{intercept_str}, {coefficients_str}'
# print(model_parameters)
