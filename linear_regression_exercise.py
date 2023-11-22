### 130Exercise - Linear Regression with scikit-learn -> New customers

# Step 1: Import necessary libraries
from sklearn.linear_model import LinearRegression
import numpy as np

# Step 2: Create the dataset
X = np.array([4, 6, 8, 10, 12, 14, 16]).reshape(-1, 1)  # Features (Cost of advertising campaign)
y = np.array([2, 2, 3, 5, 5, 6, 6])  # Target (Number of new customers)

# Step 3: Split the dataset into feature (X) and target (y) variables
# (Already done above)

# Step 4: Train a linear regression model
model = LinearRegression()
model.fit(X, y)

# Step 5: Make a prediction for the given value
prediction = model.predict(np.array([[23]]))

print(prediction)