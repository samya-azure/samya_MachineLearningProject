
# Adaboost Regression algorithm
# in this use case, predict the house price based on house size in square feet

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Step 1: Sample dataset — House Size (in sqft) vs Price
X = np.array([[500], [750], [1000], [1250], [1500], [1750], [2000], [2250], [2500], [2750]])
y = np.array([100, 150, 200, 240, 280, 310, 330, 360, 390, 420])  # in thousands

# Step 2: Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 3: Define base model and AdaBoost regressor
base_model = DecisionTreeRegressor(max_depth=3)
model = AdaBoostRegressor(estimator=base_model, n_estimators=50, learning_rate=0.8, random_state=42)

# Step 4: Train the model
model.fit(X_train, y_train)

# Step 5: Predict on test set
y_pred = model.predict(X_test)

# Step 6: Evaluation
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R^2 Score:", r2_score(y_test, y_pred))

# Step 7: Predict on new values
X_input = np.array([[1200], [1600], [2100]])
predicted_prices = model.predict(X_input)
print("\nPredictions on New Data:")
for sqft, price in zip(X_input.ravel(), predicted_prices):
    print(f"Size: {sqft} sqft → Predicted Price: ₹{price:.2f}K")

# Step 8: Plotting
plt.figure(figsize=(8, 6))
plt.scatter(X, y, color='red', label='Actual Data')
plt.scatter(X_test, y_pred, color='blue', label='Predicted Test Data')
plt.plot(X, model.predict(X), color='green', label='Regression Curve')
plt.xlabel("House Size (sqft)")
plt.ylabel("Price (₹ in thousands)")
plt.title("AdaBoost Regression Example")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
