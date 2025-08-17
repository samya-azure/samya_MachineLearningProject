
# in this Decision Tree Regression , the datasets are get first splits based on certain 
# condition.The process is, calculate the Gini Impurity of of that two groups
# the group has lowest Gini will be treated as deciding split logic

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Step 1: Sample Dataset
# Each row represents: [Income, Credit Score]
X = np.array([
    [60000, 720],
    [60000, 680],
    [40000, 750],
    [52000, 710],
    [45000, 690],
    [70000, 730]
])

# 1 = Approve, 0 = Deny
y = np.array([1, 0, 0, 1, 0, 1])

# Step 2: Create and train the Decision Tree model
model = DecisionTreeClassifier()
model.fit(X, y)

# Step 3: Predict a new applicant
new_applicant = [[60000, 720]]
prediction = model.predict(new_applicant)

print("Prediction for applicant with income 60,000 and credit score 720:")
print("Loan Approved!" if prediction[0] == 1 else "Loan Denied!")

# Step 4: Visualize the Decision Tree
plt.figure(figsize=(10, 6))
plot_tree(
    model,
    feature_names=["Income", "Credit Score"],
    class_names=["Deny", "Approve"],
    filled=True,
    rounded=True
)
plt.title("Decision Tree for Loan Approval")
plt.grid(False)
plt.show()



