
import math

# --------------------------
# Step 1: Inputs (features)
# --------------------------
hot_weather = 1.0    # Very hot
hunger = 0.5         # Somewhat hungry

# Combine inputs into a list
inputs = [hot_weather, hunger]

# --------------------------
# Step 2: Weights
# --------------------------
# Assign importance to each input
weights = [2.0, 0.5]   # Hot weather is more important than hunger

# --------------------------
# Step 3: Bias
# --------------------------
bias = -2.0  # Hard to convince (need strong reason to eat ice cream)

# --------------------------
# Step 4: Weighted sum (z)
# --------------------------
z = (inputs[0] * weights[0]) + (inputs[1] * weights[1]) + bias
print(f"Weighted Sum (z): {z:.2f}")

# --------------------------
# Step 5: Activation using Sigmoid
# --------------------------
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

output = sigmoid(z)
print(f"Output after Sigmoid: {output:.3f}")

# --------------------------
# Step 6: Interpret the Output
# --------------------------
if output > 0.5:
    print("Decision: Eat Ice Cream")
else:
    print("Decision: Don't Eat Ice Cream")
