
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor

# Sample housing dataset
data = {
    'Bedrooms': [2, 3, 4, 3, 4, 5, 1, 2, 3, 5],
    'Bathrooms': [1, 2, 3, 2, 3, 4, 1, 1, 2, 4],
    'Size': [850, 1200, 1800, 1400, 2000, 2200, 600, 750, 1100, 2500],
    'Age': [10, 5, 2, 7, 3, 1, 15, 12, 6, 1],
    'Price': [150000, 200000, 320000, 230000, 340000, 400000, 100000, 120000, 180000, 450000]
}

df = pd.DataFrame(data)

# Feature and target selection
feature_names = ['Bedrooms', 'Bathrooms', 'Size', 'Age']
X = df[feature_names]
y = df['Price']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_names)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_names)

# Train LightGBM regressor
model = LGBMRegressor()
model.fit(X_train_scaled, y_train)

# Predict on test data
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error on Test Set: {mse:.2f}")

# --- User Input ---
print("\nEnter details to predict house price:")
bedrooms = int(input("Enter number of bedrooms: "))
bathrooms = int(input("Enter number of bathrooms: "))
size = float(input("Enter size (in sqft): "))
age = int(input("Enter age of house (in years): "))

# Prepare user input as DataFrame
user_input = pd.DataFrame([[bedrooms, bathrooms, size, age]], columns=feature_names)

# Standardize user input and preserve feature names
user_input_scaled = pd.DataFrame(scaler.transform(user_input), columns=feature_names)

# Predict
predicted_price = model.predict(user_input_scaled)
print(f"\nPredicted House Price: ₹{predicted_price[0]:,.2f}")
