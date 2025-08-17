
# Adaboost Classification algorithm
# predict the person buy the product or not based on their age and salary

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Step 1: Create dataset
data = {
    'Age': [22, 25, 47, 52, 46, 56, 55, 60, 34, 43],
    'Salary': [20000, 25000, 47000, 52000, 48000, 60000, 58000, 62000, 30000, 43000],
    'Buy':    [0,     0,     1,     1,     1,     1,     1,     1,     0,     1]
}

df = pd.DataFrame(data)
X = df[['Age', 'Salary']]
y = df['Buy']

# Step 2: Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 3: AdaBoost model
base_model = DecisionTreeClassifier(max_depth=1)
model = AdaBoostClassifier(estimator=base_model, n_estimators=10, learning_rate=1.0)

model.fit(X_train, y_train)

# Step 4: Predictions
y_pred = model.predict(X_test)

# Step 5: Print evaluation
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))

# Step 6: Print predictions
print("\nPredictions:")
for i in range(len(X_test)):
    print(f"Age: {X_test.iloc[i]['Age']}, Salary: {X_test.iloc[i]['Salary']} => Predicted Buy: {y_pred[i]}")

# Step 7: Plot feature importance
importances = model.feature_importances_
plt.figure(figsize=(6, 4))
plt.bar(['Age', 'Salary'], importances, color='skyblue')
plt.title("Feature Importance (AdaBoost)")
plt.ylabel("Importance Score")
plt.grid(True)
plt.tight_layout()
plt.show()

# Step 8: Visualize decision boundary (2D plot)
from matplotlib.colors import ListedColormap

# Only for visualization — use mesh grid
x_min, x_max = X['Age'].min() - 5, X['Age'].max() + 5
y_min, y_max = X['Salary'].min() - 5000, X['Salary'].max() + 5000
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))

Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(8, 6))
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA'])
plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.5)
plt.scatter(X_train['Age'], X_train['Salary'], c=y_train, edgecolors='k', label='Train')
plt.scatter(X_test['Age'], X_test['Salary'], c=y_test, edgecolors='k', marker='x', label='Test')
plt.title("AdaBoost Classification Decision Boundary")
plt.xlabel("Age")
plt.ylabel("Salary")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
