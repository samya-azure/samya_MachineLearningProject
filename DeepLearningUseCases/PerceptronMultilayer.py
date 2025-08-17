
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# ----------------------------
# Step 1: Define Dataset (balanced a bit more)
# ----------------------------
X = np.array([
    [1, 1, 1],  # valid - repeat to help learning
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 0],
    [1, 0, 1],
    [0, 1, 1],
    [0, 0, 1],
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 0]
])

y = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0])  # Now 4 positive examples

# ----------------------------
# Step 2: Build MLP Model
# ----------------------------
model = Sequential()
model.add(Dense(6, input_dim=3, activation='relu'))  # More neurons
model.add(Dense(4, activation='relu'))               # Add a hidden layer
model.add(Dense(1, activation='sigmoid'))            # Output layer

# ----------------------------
# Step 3: Compile the Model
# ----------------------------
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# ----------------------------
# Step 4: Train the Model (more epochs)
# ----------------------------
model.fit(X, y, epochs=500, verbose=0)

# ----------------------------
# Step 5: Test on New Data
# ----------------------------
new_data = np.array([
    [1, 1, 1],  # should grant access
    [1, 0, 1],
    [0, 1, 0],
    [1, 1, 0]
])

predictions = model.predict(new_data)

# ----------------------------
# Step 6: Show Results
# ----------------------------
print("\nFinal MLP Door Access Results:\n")
for sample, prob in zip(new_data, predictions):
    result = "Access GRANTED" if prob >= 0.5 else "Access DENIED"
    print(f"Input: {sample.tolist()} => Probability: {prob[0]:.2f} => {result}")
