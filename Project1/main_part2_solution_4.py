import os
import sys

import requests

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error as mape

# checking ../Data directory presence
if not os.path.exists('../Data'):
    os.mkdir('../Data')

# download data if it is unavailable
if 'data.csv' not in os.listdir('../Data'):
    url = "https://www.dropbox.com/s/3cml50uv7zm46ly/data.csv?dl=1"
    r = requests.get(url, allow_redirects=True)
    open('../Data/data.csv', 'wb').write(r.content)

# read data
data = pd.read_csv('../Data/data.csv')
X = pd.DataFrame(data['rating'])
y = data['salary']


best_mape = sys.maxsize

for i in range(2, 5):
    x_copy = X.copy()
    x_copy['rating'] = x_copy['rating'] ** i
    x_train, x_test, y_train, y_test = train_test_split(x_copy, y, test_size=0.3, random_state=100)
    linear_regression = LinearRegression()
    
    linear_regression.fit(x_train, y_train)
    y_pred = linear_regression.predict(x_test)
    mape_score = mape(y_test, y_pred)
    
    if mape_score < best_mape:
        best_mape = mape_score

print(round(best_mape, 5))