import joblib
import numpy as np
import pandas as pd

model = joblib.load("models/california_linear_regression.joblib")
scaler = joblib.load("models/scaler_housing.joblib")

# Example new data (replace with real features)
feature_names = ["longitude","latitude","housing_median_age","total_rooms",
                 "total_bedrooms","population","households","median_income"]

new_data = pd.DataFrame([[ -122.23, 37.88, 41.0, 880.0, 129.0, 322.0, 126.0, 8.3252 ]],
                        columns=feature_names)

new_data_scaled = scaler.transform(new_data)

# Make prediction price
predicted_price = model.predict(new_data_scaled)
print(predicted_price)