
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 1. Sample dataset
data = {
    'Area': [1000, 1200, 1500, 1800, 2000, 2200, 2500, 2700, 3000, 3200],
    'Bedrooms': [2, 2, 3, 3, 4, 4, 4, 5, 5, 5],
    'Age': [5, 7, 10, 3, 2, 1, 4, 8, 6, 3],
    'Price': [100, 120, 150, 180, 200, 220, 250, 270, 300, 320]  # in Lakhs
}
df = pd.DataFrame(data)

# 2. Features and target
X = df[['Area', 'Bedrooms', 'Age']]
y = df['Price']

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train the XGBoost Regressor
model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

# 5. Predict test set and evaluate
y_pred = model.predict(X_test)
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error (RMSE):", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R-squared Score:", r2_score(y_test, y_pred))

# 6. Plot actual vs predicted
plt.figure(figsize=(8,5))
plt.scatter(y_test, y_pred, color='blue', label='Predicted vs Actual')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Ideal Fit')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("XGBoost Regression - Actual vs Predicted Prices")
plt.legend()
plt.grid(True)
plt.show()

# 7. User Input for Prediction
print("\nEnter details to predict house price:")
area = float(input("Enter area (in sq ft): "))
bedrooms = int(input("Enter number of bedrooms: "))
age = int(input("Enter age of the house (in years): "))

# Prepare input and predict
user_input = pd.DataFrame([[area, bedrooms, age]], columns=['Area', 'Bedrooms', 'Age'])
predicted_price = model.predict(user_input)[0]
print(f"\nPredicted House Price: ₹{predicted_price:.2f} Lakhs")
