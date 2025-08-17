
from sklearn.tree import DecisionTreeRegressor
import numpy as np
import matplotlib.pyplot as plt

# Input data: years of experience and output is salary 
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
y = np.array([30000, 35000, 40000, 45000, 50000, 60000, 70000, 80000, 90000, 100000])

# Train the model
model = DecisionTreeRegressor()
model.fit(X, y)

# Predict salary for 6.5 years of experience
print(model.predict([[6.5]]))  # May return something like 60000

# Optional: visualize it
X_test = np.arange(1, 11, 0.1).reshape(-1, 1)
y_pred = model.predict(X_test)

plt.scatter(X, y, color='red', label='Original Data')
plt.plot(X_test, y_pred, color='blue', label='Decision Tree Prediction')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Decision Tree Regression')
plt.grid(True)
plt.legend()
plt.show()
