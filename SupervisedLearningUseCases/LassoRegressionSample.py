
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso

# Step 1: Create multi-feature dataset
# Let's say we have 7 students
# Features: [hours studied, number of breaks, sleep hours, random noise]

X = np.array([
    [1, 3, 7, 15],
    [2, 3, 7.5, 12],
    [3, 2, 6.5, 11],
    [4, 2, 6, 9],
    [5, 1, 5.5, 8],
    [6, 1, 5, 6],
    [7, 1, 4.5, 5]
])

# Target: Marks scored
y = np.array([50, 58, 65, 70, 75, 78, 80])

# Step 2: Train Lasso model
lasso = Lasso(alpha=1.0)  # Try changing alpha to see more/less elimination
lasso.fit(X, y)

# Step 3: Show coefficients
features = ['Hours Studied', 'Breaks', 'Sleep Hours', 'Random Noise']
print("Coefficients (Feature Importance):")
for name, coef in zip(features, lasso.coef_):
    status = "Eliminated" if coef == 0 else "Kept"
    print(f"{name}: {coef:.3f} ({status})")

# Step 4: Predict for new students (test data)
X_test = np.array([
    [1, 3, 7, 15],
    [2, 2, 6.8, 13],
    [3, 2, 6, 10],
    [4, 2, 5.8, 9],
    [5, 1, 5.2, 7],
    [6, 1, 5, 6],
    [7, 1, 4.5, 5]
])
predicted = lasso.predict(X_test)

# Step 5: Print predictions
print("\nPredictions (Marks Scored):")
for i, (features_set, pred) in enumerate(zip(X_test, predicted)):
    print(f"Student {i+1}: Features={features_set} → Predicted Marks: {pred:.2f}")

# Step 6: Plotting (Hours Studied vs Predicted Marks)
hours_studied = X_test[:, 0]  # Extracting only the first feature for X-axis
plt.figure(figsize=(8, 5))
plt.plot(hours_studied, predicted, marker='o', color='blue', label='Lasso Prediction')
plt.scatter(X[:, 0], y, color='red', label='Training Data')
plt.title("Lasso Regression: Hours Studied vs Predicted Marks")
plt.xlabel("Hours Studied")
plt.ylabel("Marks Scored")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
