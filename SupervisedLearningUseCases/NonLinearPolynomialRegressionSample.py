# This example is based on Non-Linear Regression (Polynomial)
# In the below sample, x: Dosage of medicine, y: Improvement
# The sample dataset below will be treated as non-linear sample, because, here we can see 
# this looks like a curve — fast improvement at first, then slows down.
# This is Polynomial Degree 2

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Step 1: Prepare data
x = np.array([1, 2, 3, 4, 5, 6, 7]).reshape(-1, 1)
y = np.array([15, 40, 60, 70, 75, 77, 78])

# Step 2: Transform input to polynomial features (degree 2)
poly = PolynomialFeatures(degree=2)
x_poly = poly.fit_transform(x)

# Step 3: Train model
model = LinearRegression()
model.fit(x_poly, y)

# Step 4: Predict and visualize
x_fit = np.linspace(1, 7, 100).reshape(-1, 1)
x_fit_poly = poly.transform(x_fit)
y_fit = model.predict(x_fit_poly)

# Plotting
plt.scatter(x, y, color='red', label='Original Data')
plt.plot(x_fit, y_fit, color='blue', label='Polynomial Curve (Non-linear)')
plt.xlabel('Dosage')
plt.ylabel('Improvement')
plt.title('Non-Linear Regression Example')
plt.legend()
plt.grid(True)
plt.show()
