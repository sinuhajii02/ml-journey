# A linear regression to predict housing prices using the Boston Housing

# Import necessary packages
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib


# Load example data set
df = pd.read_csv('data/housing.csv')
# We are going to test for the median house value column
X = df.drop(columns=["median_house_value", "ocean_proximity"])
y = df["median_house_value"]
X = X.fillna(X.mean())

# Scale dataset
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data train/test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate
y_prediction = model.predict(X_test)
mean_squared_error = mean_squared_error(y_test, y_prediction)
r2 = r2_score(y_test, y_prediction)
print(f"Test MSE: {mean_squared_error:.2f}")
print(f"Test R2: {r2:.2f}")

joblib.dump(model, "models/california_linear_regression.joblib")
joblib.dump(scaler, "models/scaler_housing.joblib")
print("Saved")

