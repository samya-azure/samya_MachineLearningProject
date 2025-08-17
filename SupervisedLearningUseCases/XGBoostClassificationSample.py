
# Step 1: Install xgboost if not already installed
# We will classify houses into price ranges:
# ( 0 → Low price (e.g., < ₹50 lakhs))
# (1 → Medium price (₹50–₹100 lakhs))
# (2 → High price (> ₹100 lakhs))

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Simulated dataset
data = {
    'area_sqft': [800, 1200, 1500, 600, 2000, 2500, 3000, 1000, 1800, 900, 1700, 1300],
    'bedrooms':  [2, 3, 3, 1, 4, 4, 5, 2, 3, 2, 3, 3],
    'bathrooms': [1, 2, 2, 1, 3, 3, 4, 1, 2, 1, 2, 2],
    'age_years': [10, 5, 2, 15, 1, 3, 2, 12, 4, 8, 5, 6],
    'price_range': [0, 1, 1, 0, 2, 2, 2, 0, 1, 0, 1, 1]
}

df = pd.DataFrame(data)

# Features and label
X = df[['area_sqft', 'bedrooms', 'bathrooms', 'age_years']]
y = df['price_range']  # 0: Low, 1: Moderate, 2: High

# Split dataset with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

# Train XGBoost Classifier
model = XGBClassifier(objective='multi:softmax', num_class=3, eval_metric='mlogloss', random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=1))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Low', 'Moderate', 'High'],
            yticklabels=['Low', 'Moderate', 'High'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
