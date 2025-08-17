
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Sample dataset (Bedrooms, Size (sqft), Location score, House age) → Price (in lakhs)
X = np.array([
    [2, 800, 6, 10],
    [3, 1200, 8, 5],
    [4, 1500, 9, 3],
    [3, 1000, 7, 8],
    [5, 2000, 10, 2],
    [2, 850, 6, 12],
    [3, 950, 7, 9],
    [4, 1700, 8, 4],
    [3, 1100, 7, 6],
    [2, 750, 5, 15]
])
y = np.array([40, 65, 90, 55, 120, 42, 50, 95, 60, 35])  # Prices in lakhs

# Step 2: Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

# Step 4: Train Gradient Boosting Regressor
model = GradientBoostingRegressor(
    n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42
)
model.fit(X_train, y_train)

# Step 5: Predict on test data
y_pred = model.predict(X_test)

# Step 6: Print model performance
print("Actual Prices   :", y_test)
print("Predicted Prices:", np.round(y_pred, 2))
print("MSE:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# Step 7: User input for prediction
print("\nPredict House Price from Your Input:")
try:
    bedrooms = int(input("Enter number of bedrooms: "))
    size = float(input("Enter house size in sqft: "))
    location_score = float(input("Enter location score (1–10): "))
    age = float(input("Enter house age in years: "))

    user_input = np.array([[bedrooms, size, location_score, age]])
    user_scaled = scaler.transform(user_input)
    predicted_price = model.predict(user_scaled)[0]

    print(f"\nPredicted House Price: ₹{predicted_price:.2f} lakhs")

except:
    print("Invalid input. Please enter valid numeric values.")

# Step 8: Plotting actual vs predicted prices
plt.figure(figsize=(8, 5))
plt.plot(range(len(y_test)), y_test, 'ro-', label='Actual Price')
plt.plot(range(len(y_pred)), y_pred, 'bo--', label='Predicted Price')
plt.xlabel("Test Sample Index")
plt.ylabel("Price (in lakhs)")
plt.title("Actual vs Predicted House Prices")
plt.legend()
plt.grid(True)
plt.show()
