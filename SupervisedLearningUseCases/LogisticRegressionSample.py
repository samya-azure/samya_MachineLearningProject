# Logistic Regression used for Binary Classfication (yes/no, 1/0, true/false)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# x = study hours by student
# Step 1: Prepare the data
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])  # 0 = Fail, 1 = Pass

# Step 2: Train the logistic regression model
model = LogisticRegression()
model.fit(x, y)

# Step 3: Predict probability for a specific value
hours = 6
prob = model.predict_proba([[hours]])[0][1]
prediction = model.predict([[hours]])[0]

print(f"Predicted probability of passing for {hours} hours studied: {prob:.4f}")
print(f"Predicted class (0 = Fail, 1 = Pass): {prediction}")

# Step 4: Plot sigmoid curve
x_test = np.linspace(0, 11, 100).reshape(-1, 1)
y_prob = model.predict_proba(x_test)[:, 1]

plt.figure(figsize=(8, 5))
plt.scatter(x, y, color='red', label='Actual Data')
plt.plot(x_test, y_prob, color='blue', label='Sigmoid Curve')
plt.axhline(0.5, color='green', linestyle='--', label='Decision Boundary (0.5)')
plt.axvline(hours, color='orange', linestyle='--', label=f'{hours} Hours Input')
plt.xlabel('Hours Studied')
plt.ylabel('Probability of Passing')
plt.title('Logistic Regression - Pass Prediction')
plt.legend()
plt.grid(True)
plt.show()
