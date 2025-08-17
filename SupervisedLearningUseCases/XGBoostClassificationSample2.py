
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Sample dataset
data = {
    'size': [1000, 1500, 800, 1200, 2000, 2500, 600, 1100, 1800, 3000],
    'bedrooms': [2, 3, 2, 3, 4, 4, 1, 2, 3, 5],
    'location_score': [7, 8, 6, 7, 9, 10, 5, 6, 9, 10],
    'price': [200000, 300000, 150000, 250000, 500000, 600000, 120000, 220000, 450000, 750000]
}
df = pd.DataFrame(data)

# 2. Convert price into categories (Low, Medium, High)
def price_category(price):
    if price < 250000:
        return 0  # Low
    elif price < 500000:
        return 1  # Medium
    else:
        return 2  # High

df['price_category'] = df['price'].apply(price_category)

# 3. Features and target
X = df[['size', 'bedrooms', 'location_score']]
y = df['price_category']

# 4. Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 5. Train XGBoost Classifier
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
model.fit(X_train, y_train)

# 6. Predictions
y_pred = model.predict(X_test)

# 7. Evaluation
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))

# 8. Feature Importance Plot
xgb.plot_importance(model)
plt.title("Feature Importance")
plt.show()

# 9. Predict on new input
new_data = pd.DataFrame({'size': [1600], 'bedrooms': [3], 'location_score': [8]})
predicted_class = model.predict(new_data)[0]
price_labels = {0: 'Low', 1: 'Medium', 2: 'High'}
print(f"Predicted house price category: {price_labels[predicted_class]}")
