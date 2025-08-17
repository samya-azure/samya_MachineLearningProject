# Random Forest Regression algorithm is like a team Decision Trees working together to make
# better prediction. It predicts continuous/numeric values

from sklearn.ensemble import RandomForestRegressor
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Sample data (Hours studied vs Marks scored)
x = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9]])
y = np.array([50, 55, 58, 62, 68, 70, 72, 75, 80])

# Step 2: Create and train the Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(x, y)

# Step 3: Create test input to predict (between 1 and 9 in small steps)
x_test = np.linspace(1, 9, 100).reshape(-1, 1)

# Step 4: Predict the output using the trained model
y_pred = model.predict(x_test)

# Step 5: Print a few sample predictions
print("Predicted marks for different hours studied:")
for i in range(0, 100, 10):  # print every 10th point
    print(f"Hours: {x_test[i][0]:.1f} → Predicted Marks: {y_pred[i]:.2f}")

# Step 6: Plot original data and prediction curve
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='red', label='Actual Data (Hours vs Marks)', s=60)
plt.plot(x_test, y_pred, color='green', label='Random Forest Prediction Curve', linewidth=2)
plt.xlabel('Hours Studied')
plt.ylabel('Marks Scored')
plt.title('Random Forest Regression - Hours Studied vs Marks')
plt.legend()
plt.grid(True)
plt.show()
