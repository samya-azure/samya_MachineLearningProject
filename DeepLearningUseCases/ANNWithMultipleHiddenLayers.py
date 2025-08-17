
# This example is based on dataset which has 10000 rows
# Here use 3 Hidden Layers with 16,8 and 4 neurons with ReLu
# Prediction for age and salary input is below in the code

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# -------------------------------
# Step 1: Load CSV File
# -------------------------------
# Replace with the actual path to your CSV
file_path = "./DataSets/synthetic_customer_data.csv"  # If in same directory

# Load dataset
df = pd.read_csv(file_path)

# -------------------------------
# Step 2: Prepare Features & Labels
# -------------------------------
X = df[['Age', 'Salary']].values
y = df['Buy'].values

# Scale Age and Salary to 0–1 range
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------
# Step 3: Split into Train/Test
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# -------------------------------
# Step 4: Build ANN with Multiple Hidden Layers
# -------------------------------
model = Sequential()
model.add(Dense(16, input_dim=2, activation='relu'))  # Hidden layer 1
model.add(Dense(8, activation='relu'))                # Hidden layer 2
model.add(Dense(4, activation='relu'))                # Hidden layer 3
model.add(Dense(1, activation='sigmoid'))             # Output layer

# -------------------------------
# Step 5: Compile & Train
# -------------------------------
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print("Training model...\n")
model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)
print("Training complete!")

# -------------------------------
# Step 6: Evaluate Model
# -------------------------------
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\nModel Accuracy on Test Set: {accuracy:.4f}")

# -------------------------------
# Step 7: Predict for a New Customer
# -------------------------------
# Example input (Age = 40, Salary = 70000)
#new_customer = np.array([[40, 70000]])
new_customer = np.array([[40, 130000]])
new_customer_scaled = scaler.transform(new_customer)
prediction = model.predict(new_customer_scaled)

print(f"\nPrediction Probability: {prediction[0][0]:.3f}")
if prediction[0][0] > 0.5:
    print("The customer is likely to BUY.")
else:
    print("The customer is NOT likely to buy.")
