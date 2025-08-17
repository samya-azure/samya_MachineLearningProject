
# Perceptron model
# Door Access Control system

import numpy as np

# ----------------------------
# Step 1: Define Dataset
# ----------------------------
X = np.array([
    [1, 1, 1],
    [1, 1, 0],
    [1, 0, 1],
    [0, 1, 1],
    [0, 0, 1],
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 0]
])

y = np.array([1, 0, 0, 0, 0, 0, 0, 0])  # Only [1,1,1] should return 1

# ----------------------------
# Step 2: Initialize Parameters
# ----------------------------
weights = np.zeros(X.shape[1])  # [0, 0, 0]
bias = 0
learning_rate = 0.2
epochs = 20  # Increased epochs for better learning

# ----------------------------
# Step 3: Training Loop
# ----------------------------
print("Training Perceptron:\n")

for epoch in range(epochs):
    for inputs, target in zip(X, y):
        z = np.dot(inputs, weights) + bias
        prediction = 1 if z >= 0 else 0
        error = target - prediction

        # Update weights and bias
        weights += learning_rate * error * inputs
        bias += learning_rate * error

# ----------------------------
# Step 4: Test on New Data
# ----------------------------
new_data = np.array([
    [1, 1, 1],  # Should grant access
    [1, 0, 1],  # Should deny
    [0, 1, 0],  # Should deny
    [1, 1, 0]   # Should deny
])

print("Final Testing Results:\n")
for sample in new_data:
    z = np.dot(sample, weights) + bias
    prediction = 1 if z >= 0 else 0
    status = "Access GRANTED" if prediction == 1 else "Access DENIED"
    print(f"Input: {sample.tolist()} => {status}")
