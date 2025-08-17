
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA  # ARIMA can act like ARMA if d=0
import warnings
warnings.filterwarnings("ignore")

# Generate stationary time series data (white noise with ARMA structure)
np.random.seed(42)
n = 200
# Create ARMA process manually: AR(2) + MA(1)
noise = np.random.normal(0, 1, n)
data = pd.Series([0]*n)

# AR(2): Y_t = 0.5 * Y_{t-1} - 0.25 * Y_{t-2} + noise + 0.3 * noise_{t-1}
for t in range(2, n):
    data[t] = 0.5*data[t-1] - 0.25*data[t-2] + noise[t] + 0.3*noise[t-1]

# Plot the data
plt.figure(figsize=(10, 4))
plt.plot(data, label="Simulated ARMA(2,1) Data")
plt.title("Stationary Time Series (ARMA)")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.show()

# Fit ARMA model using ARIMA with d=0
# ARMA(p, q) = ARIMA(p, 0, q)
model = ARIMA(data, order=(2, 0, 1))  # AR=2, MA=1, d=0
result = model.fit()

# Forecast next 10 points
forecast = result.forecast(steps=10)

# Plot forecast
plt.figure(figsize=(10, 4))
plt.plot(data, label="Observed")
plt.plot(range(len(data), len(data)+10), forecast, label="Forecast", color='red')
plt.title("ARMA Forecast")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.show()

# Summary of the model
print(result.summary())
