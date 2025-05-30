from sklearn.model_selection import cross_val_score, cross_val_predict, KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np

# Input features and target
X = model_data[["Trips 25-50 Miles"]]
y = model_data["Number of Trips 5-10"]  # Use Series, not DataFrame for y in CV

# Set up the model and 5-fold CV
model = LinearRegression()
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Cross-validated predictions
y_pred_cv = cross_val_predict(model, X, y, cv=cv)

# Cross-validated evaluation metrics
r2_cv = r2_score(y, y_pred_cv)
mse_cv = mean_squared_error(y, y_pred_cv)
mae_cv = mean_absolute_error(y, y_pred_cv)
rmse_cv = np.sqrt(mse_cv)

# Print results
print("Cross-Validation Evaluation Metrics (5-Fold):")
print(f"R-squared: {r2_cv:.4f}")
print(f"Mean Squared Error: {mse_cv:,.2f}")
print(f"Root Mean Squared Error: {rmse_cv:,.2f}")
print(f"Mean Absolute Error: {mae_cv:,.2f}")
