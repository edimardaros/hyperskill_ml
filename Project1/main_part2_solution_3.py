import os
import numpy as np
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

# write your code here
mape_results = []
for power in range(2, 5):
    predictor = data['rating'].pow(power)
    X = np.array(predictor).reshape(-1, 1)
    y = data['salary']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_predict = model.predict(X_test)

    mape_value = mape(y_test, y_predict)

    mape_results.append(round(mape_value, 5))

print(min(mape_results))