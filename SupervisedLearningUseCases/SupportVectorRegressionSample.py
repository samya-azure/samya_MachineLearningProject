# Support Vector Regression algorithm is the regression version of Support Vector Machine
# It tries to find a line or curve , that best fit the data, but with a margin of tolerance
# this margin of tolerance is called, Epsilon


import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR

# Step 1: Prepare the data
x = np.array([1, 2, 3, 4, 5, 6, 7]).reshape(-1, 1)
y = np.array([50, 60, 65, 70, 75, 78, 80])

# Step 2: Create SVR model with epsilon
epsilon_value = 2
model = SVR(kernel='rbf', C=100, epsilon=epsilon_value)
model.fit(x, y)

# Step 3: Predict marks
x_test = np.linspace(1, 7, 12).reshape(-1, 1)
y_pred = model.predict(x_test)

# Step 4: Print predictions
print(f"Predicted marks for different hours studied (with ε = {epsilon_value}):")
for hour, mark in zip(x_test.flatten(), y_pred):
    print(f"Hours: {hour:.1f} → Predicted Marks: {mark:.2f}")

# Step 5: Plotting
plt.scatter(x, y, color='red', label='Original Data')
plt.plot(x_test, y_pred, color='blue', label='SVR Prediction')

# Plot epsilon margin (±ε tube)
plt.plot(x_test, y_pred + epsilon_value, 'k--', label=f'+ε = {epsilon_value}')
plt.plot(x_test, y_pred - epsilon_value, 'k--', label=f'-ε = -{epsilon_value}')

plt.xlabel('Hours Studied')
plt.ylabel('Marks Scored')
plt.title('Support Vector Regression with Epsilon Margin')
plt.legend()
plt.grid(True)
plt.show()
