# KNN algorithm predicts the output based on the visiting neartest neighbours/data points

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# Step 1: Dataset - Hours Studied vs Passed (1) or Failed (0)
x = np.array([[1], [2], [3], [4], [5], [6]])
y = np.array([0, 0, 0, 1, 1, 1])

# Step 2: Train the KNN model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(x, y)

# Step 3: Predict for a range of values (for plotting)
x_test = np.linspace(0, 7, 100).reshape(-1, 1)
y_pred = model.predict(x_test)

# Step 4: Predict for a specific case, e.g., 4.5 hours
custom_hour = np.array([[4.5]])
custom_prediction = model.predict(custom_hour)
print(f"Prediction for {custom_hour[0][0]} hours: {'Pass' if custom_prediction[0] == 1 else 'Fail'}")

# Step 5: Plotting
plt.figure(figsize=(10, 6))

# Original data points
plt.scatter(x, y, color='blue', label='Original Data (0=Fail, 1=Pass)', s=100)

# Decision boundary line
plt.plot(x_test, y_pred, color='green', label='KNN Prediction Boundary')

# Highlight prediction
plt.scatter(custom_hour, custom_prediction, color='red', s=200, edgecolors='black', label=f'Prediction for {custom_hour[0][0]} hrs')

plt.title('K-Nearest Neighbors (KNN) Classification')
plt.xlabel('Hours Studied')
plt.ylabel('Result (0=Fail, 1=Pass)')
plt.yticks([0, 1])
plt.legend()
plt.grid(True)
plt.show()
