# Door Access Control system Use Case
# Business Logic:
    # Doors open only if,
        # 1. Has ID Card = 1 , (means the person has ID card)
        # 2. Knows PIN = 1, (means, the person knows the correct PIN)
        # 3. Not blacklisted = 1, (means, the person is not blacklisted)
    # so, all must be true (1), i.e., similar to AND logic gate

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ----------------------------------
# Step 1: Define Dataset
# ----------------------------------
# [Has ID, Knows PIN, Not Blacklisted]
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

# Labels: 1 = Access Granted, 0 = Access Denied
y = np.array([
    1, 0, 0, 0, 0, 0, 0, 0
])

# ----------------------------------
# Step 2: Split Dataset
# ----------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# ----------------------------------
# Step 3: Define MCP Neuron
# ----------------------------------
def mcp_neuron(inputs, weights, threshold):
    total = np.dot(inputs, weights)  # weighted sum
    return 1 if total >= threshold else 0

# ----------------------------------
# Step 4: Train Model (simulate training by fixed weights)
# ----------------------------------
# We fix weights and threshold based on logic:
weights = np.array([1, 1, 1])
threshold = 3

# ----------------------------------
# Step 5: Test Model
# ----------------------------------
y_pred = []
for sample in X_test:
    output = mcp_neuron(sample, weights, threshold)
    y_pred.append(output)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.2f}")

# ----------------------------------
# Step 6: Predict New Input
# ----------------------------------
new_data = np.array([
    [1, 1, 1],  # Expected: Access Granted
    [1, 0, 1],  # Expected: Denied
    [0, 1, 0],  # Expected: Denied
])

print("\nPredicting Access for New Data:")
for sample in new_data:
    result = mcp_neuron(sample, weights, threshold)
    access = "Access GRANTED" if result == 1 else "Access DENIED"
    print(f"Input: {sample.tolist()} => {access}")
