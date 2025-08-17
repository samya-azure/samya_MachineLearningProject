
from sklearn.linear_model import LinearRegression

# Input features: [hours studied, practice tests taken]
x = [
    [1, 1],
    [2, 1],
    [3, 2],
    [4, 2],
    [5, 3]
]

# Output target: Marks scored
y = [52, 58, 65, 70, 78]

# Create and train the model
model = LinearRegression()
model.fit(x, y)

# Predict marks for a student who studied 6 hours and took 3 practice tests
predicted = model.predict([[6, 3]])
print("Predicted Marks:", predicted[0])
