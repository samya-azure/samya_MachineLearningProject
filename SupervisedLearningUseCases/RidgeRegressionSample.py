# Ridge Regression is just Linear Regression with a penalty on big coefficients.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge

# Data: Hours studied vs Marks scored
x = np.array([1, 2, 3, 4, 5, 6, 7]).reshape(-1, 1)
y = np.array([50, 60, 65, 70, 75, 78, 80])

# Ridge Regression Model with regularization strength alpha (λ)
model = Ridge(alpha=1.0)
model.fit(x, y)

# Predicting for new inputs
x_test = np.linspace(1, 7, 12).reshape(-1, 1)
y_pred = model.predict(x_test)

# Print predictions
print("Predicted marks using Ridge Regression:")
for hour, mark in zip(x_test.flatten(), y_pred):
    print(f"Hours: {hour:.1f} → Predicted Marks: {mark:.2f}")

# Plot
plt.scatter(x, y, color='red', label='Original Data')
plt.plot(x_test, y_pred, color='green', label='Ridge Prediction')
plt.xlabel('Hours Studied')
plt.ylabel('Marks Scored')
plt.title('Ridge Regression Example')
plt.legend()
plt.grid(True)
plt.show()
