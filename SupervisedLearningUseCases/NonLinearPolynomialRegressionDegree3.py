# This example is based on Non-Linear Regression (Polynomial)
# In the below sample, x: Dosage of medicine, y: Improvement
# The sample dataset below will be treated as non-linear sample, because, here we can see 
# this looks like a curve — fast improvement at first, then slows down.
# This is Polynomial Degree 3

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Step 1: Dataset
x = np.array([1, 2, 3, 4, 5, 6, 7]).reshape(-1, 1)
y = np.array([15, 40, 60, 70, 75, 77, 78])

# Step 2: Use degree = 3
poly = PolynomialFeatures(degree=3)
x_poly = poly.fit_transform(x)

# Step 3: Fit model
model = LinearRegression()
model.fit(x_poly, y)

# Step 4: Predict for a smooth curve
x_fit = np.linspace(1, 7, 100).reshape(-1, 1)
x_fit_poly = poly.transform(x_fit)
y_fit = model.predict(x_fit_poly)

# Step 5: Plot
plt.scatter(x, y, color='red', label='Original Data')
plt.plot(x_fit, y_fit, color='green', label='Cubic Curve (Degree 3)')
plt.xlabel('Dosage')
plt.ylabel('Improvement')
plt.title('Polynomial Regression (Degree 3)')
plt.legend()
plt.grid(True)
plt.show()
