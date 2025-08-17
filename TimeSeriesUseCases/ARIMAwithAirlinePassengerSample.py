
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings("ignore")

# === Step 1: Load the dataset ===
file_path = r"./DataSets/airline-passengers.csv"
data = pd.read_csv(file_path)

# Convert 'Month' column to datetime and set as index
data['Month'] = pd.to_datetime(data['Month'])
data.set_index('Month', inplace=True)

# Print first few rows
print("Loaded Data:")
print(data.head())

# === Step 2: Plot the original data ===
plt.figure(figsize=(10, 5))
plt.plot(data['Passengers'], label='Monthly Passengers')
plt.title("Airline Passengers Over Time")
plt.xlabel("Year")
plt.ylabel("Passengers")
plt.grid(True)
plt.legend()
plt.show()

# === Step 3: ADF Test for Stationarity ===
result = adfuller(data['Passengers'])
print("\nAugmented Dickey-Fuller Test:")
print(f"ADF Statistic: {result[0]:.4f}")
print(f"p-value: {result[1]:.4f}")
for key, value in result[4].items():
    print(f"Critical Value ({key}): {value:.4f}")
if result[1] < 0.05:
    print("Series is likely stationary.")
else:
    print("Series is likely non-stationary. Applying differencing (d=1).")

# === Step 4: Fit ARIMA model ===
print("\nFitting ARIMA(2,1,2) model...")
model = ARIMA(data['Passengers'], order=(2, 1, 2))  # (p, d, q)
model_fit = model.fit()
print("Model fitted successfully.")

# Print model summary
print("\nModel Summary:")
print(model_fit.summary())

# === Step 5: Forecast the next 12 months ===
forecast = model_fit.forecast(steps=12)

print("\nForecasted Values for Next 12 Months:")
print(forecast)

# Create future date index
future_dates = pd.date_range(start=data.index[-1] + pd.DateOffset(months=1), periods=12, freq='MS')

# Plot original + forecast
plt.figure(figsize=(10, 5))
plt.plot(data['Passengers'], label='Original Data')
plt.plot(future_dates, forecast, label='Forecast', color='red')
plt.title("ARIMA Forecast for Airline Passengers")
plt.xlabel("Date")
plt.ylabel("Passengers")
plt.legend()
plt.grid(True)
plt.show()
