
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# Step 1: Sample dataset (Age, Salary, Buy or Not)
X = np.array([
    [22, 25000],
    [25, 32000],
    [47, 50000],
    [52, 110000],
    [46, 59000],
    [56, 85000],
    [23, 27000],
    [51, 52000],
    [48, 41000],
    [33, 39000],
])
y = np.array([0, 0, 1, 1, 1, 1, 0, 1, 1, 0])  # 0 = Not Buy, 1 = Buy

# Step 2: Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

# Step 4: Train Gradient Boosting Classifier
model = GradientBoostingClassifier(
    n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42
)
model.fit(X_train, y_train)

# Step 5: Make predictions
y_pred = model.predict(X_test)

# Step 6: Print results
print("Predicted Labels:", y_pred)
print("Actual Labels   :", y_test)
print("Accuracy Score  :", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 7: User Input for Prediction
print("\nLet's predict for a new person:")

try:
    age = float(input("Enter Age: "))
    salary = float(input("Enter Salary: "))

    # Prepare input and scale it
    user_input = np.array([[age, salary]])
    user_scaled = scaler.transform(user_input)
    prediction = model.predict(user_scaled)[0]

    # Show result
    result = "will BUY the product " if prediction == 1 else "will NOT buy the product "
    print(f"\nPrediction: The person {result}")

except ValueError:
    print("Invalid input. Please enter valid numbers for age and salary.")

# Step 8: Plotting decision boundary
def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 100),
        np.linspace(y_min, y_max, 100)
    )
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.4, cmap='coolwarm')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k')
    plt.title("Decision Boundary - Gradient Boosting Classifier")
    plt.xlabel("Age (scaled)")
    plt.ylabel("Salary (scaled)")
    plt.grid(True)
    plt.show()

plot_decision_boundary(model, X_scaled, y)
