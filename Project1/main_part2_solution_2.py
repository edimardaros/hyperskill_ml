import os

import numpy as np
import requests

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error

np.random.seed(100)
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

# Stage 2
data.drop(['draft_round', 'age', 'experience', 'bmi'], axis=1, inplace=True)
data.dropna()
y = data['salary']
data.drop(['salary'], axis=1, inplace=True)

MAPE = []
for i in range(2, 5):
    data['power_rating'] = data['rating'].apply(lambda x: pow(x, i))
    X = data.drop(['rating'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    MAPE.append(round(mean_absolute_percentage_error(y_test, y_pred), 5))
print(min(MAPE))