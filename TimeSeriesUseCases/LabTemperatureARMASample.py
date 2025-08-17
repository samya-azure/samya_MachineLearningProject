
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore")

# Simulate lab temperature data (stationary around 25°C)
np.random.seed(42)
n = 200
noise = np.random.normal(0, 0.5, n)  # small random variation
lab_temp = pd.Series([25.0]*n)

# Generate ARMA-like process: AR(1) + MA(1)
for t in range(1, n):
    lab_temp[t] = 0.8 * lab_temp[t-1] + noise[t] + 0.3 * noise[t-1]

# Create datetime index (hourly)
dates = pd.date_range(start="2025-01-01", periods=n, freq="H")
lab_temp.index = dates

# Plot temperature data
plt.figure(figsize=(12, 4))
plt.plot(lab_temp, label="Lab Temperature (°C)")
plt.title("Simulated Lab Temperature (Stationary)")
plt.xlabel("Time")
plt.ylabel("Temperature (°C)")
plt.grid(True)
plt.legend()
plt.show()

# Fit ARMA model using ARIMA with d=0
model = ARIMA(lab_temp, order=(1, 0, 1))  # ARMA(1,1)
result = model.fit()

# Forecast next 12 hours
forecast = result.forecast(steps=12)

# Print forecasted temperatures
print("\nForecasted Lab Temperatures (Next 12 Hours):")
print(forecast.round(2).to_string())

# Plot forecast
plt.figure(figsize=(12, 4))
plt.plot(lab_temp, label="Observed Temperature")
plt.plot(pd.date_range(start=lab_temp.index[-1], periods=13, freq='H')[1:], 
         forecast, color='red', marker='o', label="Forecast")
plt.title("Lab Temperature Forecast (ARMA)")
plt.xlabel("Time")
plt.ylabel("Temperature (°C)")
plt.legend()
plt.grid(True)
plt.show()

# Print model summary
print("\nModel Summary:")
print(result.summary())
