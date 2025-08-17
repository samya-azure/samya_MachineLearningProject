
# In this example, few important points
# 2 neurons come from Input Layer (age & salary)
# In Hidden Layer, there are 4 neurons activate by ReLu
# In Output layer, 1 neuron activate Sigmoid (since we need output 1 (buy) or 0 (not buy))
# In this example, there is only 1 Hidden Layer

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# -------------------------------
# Step 1: Define Sample Dataset
# -------------------------------
# [Age, Salary] - scaled between 0 and 1
X = np.array([
    [0.2, 0.1],   # Young, low salary
    [0.3, 0.2],
    [0.4, 0.3],
    [0.5, 0.5],
    [0.6, 0.6],
    [0.7, 0.7],
    [0.8, 0.8],
    [0.9, 0.9],   # Older, high salary
    [1.0, 1.0]
])

# Target: 0 = Not Buy, 1 = Buy
y = np.array([
    0, 0, 0, 0, 1, 1, 1, 1, 1
])

# -------------------------------
# Step 2: Split into Train and Test
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("\nTraining Data:\n", X_train)
print("\nTesting Data:\n", X_test)

# -------------------------------
# Step 3: Build ANN Model
# -------------------------------
model = Sequential()
model.add(Dense(4, input_dim=2, activation='relu'))   # Hidden layer
model.add(Dense(1, activation='sigmoid'))             # Output layer

# -------------------------------
# Step 4: Compile Model
# -------------------------------
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print("\nModel compiled successfully!")

# -------------------------------
# Step 5: Train the Model
# -------------------------------
print("\nTraining the model...\n")
model.fit(X_train, y_train, epochs=100, verbose=0)
print("Training complete!")

# -------------------------------
# Step 6: Evaluate the Model
# -------------------------------
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {accuracy:.2f}")

# -------------------------------
# Step 7: Predict New Customer
# -------------------------------
new_customer = np.array([[0.65, 0.65]])  # Mid-age, mid-salary
prediction = model.predict(new_customer)

print(f"\nPrediction Probability: {prediction[0][0]:.3f}")
if prediction[0][0] > 0.5:
    print("The customer is likely to BUY the product.")
else:
    print("The customer is NOT likely to buy the product.")
