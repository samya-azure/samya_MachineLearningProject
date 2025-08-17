
# Sigmoid is working in output layer
# Relu is working in hidden layer

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# -------------------------------
# Step 1: Sample Dataset
# -------------------------------
# Features: [Age, Salary]  - Both scaled between 0 and 1
X = np.array([
    [0.2, 0.1],   # young, low salary
    [0.3, 0.2],
    [0.4, 0.3],
    [0.5, 0.5],
    [0.6, 0.6],
    [0.7, 0.7],
    [0.8, 0.8],
    [0.9, 0.9],   # older, high salary
    [1.0, 1.0]
])

# Labels: 0 = No Buy, 1 = Buy
y = np.array([
    0, 0, 0, 0, 1, 1, 1, 1, 1
])

# -------------------------------
# Step 2: Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------------
# Step 3: Build ANN with ReLU
# -------------------------------
model = Sequential()
model.add(Dense(4, input_dim=2, activation='relu'))   # Hidden layer with ReLU
model.add(Dense(1, activation='sigmoid'))             # Output layer with Sigmoid

# -------------------------------
# Step 4: Compile the Model
# -------------------------------
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# -------------------------------
# Step 5: Train the Model
# -------------------------------
model.fit(X_train, y_train, epochs=100, verbose=0)

# -------------------------------
# Step 6: Evaluate
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
