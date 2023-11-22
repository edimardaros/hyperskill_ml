from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

data = fetch_california_housing()

# Extracting the features
X = data.data
# Extracting the target attribute
Y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.8, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)
print(model.intercept_)

predictions_train = model.predict(X_train)
print(predictions_train.shape)

predictions_test = model.predict(X_test)
print(predictions_test.shape)

# ---
mse_train = mean_squared_error(y_train, predictions_train)
print(mse_train)

# 0.5179331255246699

mse_test = mean_squared_error(y_test, predictions_test)
print(mse_test)

# 0.5558915986952422

mae_train = mean_absolute_error(y_train, predictions_train)
print(mae_train)

# 0.5286283596582376

mae_test = mean_absolute_error(y_test, predictions_test)
print(mae_test)

# 0.533200130495698
