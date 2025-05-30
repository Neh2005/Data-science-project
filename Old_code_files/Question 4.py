##Question 4  - Linear Regression Model

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Load datasets
df1 = pd.read_csv("trips_by_distance.csv.csv", low_memory=False)
df2 = pd.read_csv("trips_full_data__2_.csv", low_memory=False)

# Convert Date columns to datetime
df1['Date'] = pd.to_datetime(df1['Date'], errors='coerce')
df2['Date'] = pd.to_datetime(df2['Date'], errors='coerce')

# Filter only National level data
national_only = df1[df1['Level'] == "National"].copy()
national_only['Date'] = pd.to_datetime(national_only['Date'], errors='coerce')

# Get overlapping date range from df2
start_date = df2['Date'].min()
end_date = df2['Date'].max()

# Filter both datasets for the same date range
filtered_df1 = national_only[national_only['Date'].between(start_date, end_date)]
filtered_df2 = df2[df2['Date'].between(start_date, end_date)]

# Reset index for alignment
filtered_df1 = filtered_df1.sort_values("Date").reset_index(drop=True)
filtered_df2 = filtered_df2.sort_values("Date").reset_index(drop=True)

# Prepare the model input
X = filtered_df2[["Trips 25-50 Miles"]].copy()
y = filtered_df1[["Number of Trips 5-10"]].copy()

# Drop missing values (if any)
model_data = pd.concat([X, y], axis=1).dropna()
X = model_data[["Trips 25-50 Miles"]]
y = model_data[["Number of Trips 5-10"]]

# Build Linear Regression model
model = LinearRegression()
model.fit(X, y)

# Predictions
y_pred = model.predict(X)

# Evaluation metrics
r2 = r2_score(y, y_pred)
mse = mean_squared_error(y, y_pred)
mae = mean_absolute_error(y, y_pred)

# Print model performance
print("Model Evaluation Metrics:")
print(f"R-squared: {r2:.4f}")
print(f"Mean Squared Error: {mse:.2f}")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"Intercept: {model.intercept_[0]:.2f}")
print(f"Coefficients: {model.coef_}")

# Plot
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', alpha=0.5, label='Actual Data')
plt.plot(X, y_pred, color='red', linewidth=2, label='Linear Regression Line')
plt.xlabel("Trips 25-50 Miles")
plt.ylabel("Number of Trips 5–10 Miles")
plt.title("Linear Regression: Trips 25–50 Miles vs 5–10 Miles")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# Plot: Actual vs Predicted values
plt.figure(figsize=(8, 6))
plt.scatter(y, y_pred, color='green', alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--', label='Perfect Prediction Line')
plt.xlabel("Actual Number of Trips 25–50 Miles")
plt.ylabel("Predicted Number of Trips 25–50 Miles")
plt.title("Actual vs Predicted - Linear Regression")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# Residual plot
residuals = y.values.flatten() - y_pred.flatten()
plt.figure(figsize=(10, 5))
plt.scatter(y_pred, residuals, color='purple', alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predicted Values")
plt.ylabel("Residuals (Actual - Predicted)")
plt.title("Residual Plot")
plt.grid(True)
plt.tight_layout()
plt.show()



