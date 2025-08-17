
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings("ignore")

# === Step 1: Load the dataset ===
file_path = r"./DataSets/airline-passengers.csv"
data = pd.read_csv(file_path)
data['Month'] = pd.to_datetime(data['Month'])
data.set_index('Month', inplace=True)

print("Sample Data:")
print(data.head())

# === Step 2: Plot original data ===
plt.figure(figsize=(10, 5))
plt.plot(data['Passengers'], label='Monthly Passengers')
plt.title("Monthly Airline Passengers (1949–1960)")
plt.xlabel("Date")
plt.ylabel("Passengers")
plt.grid(True)
plt.legend()
plt.show()

# === Step 3: Fit SARIMA model ===
# We'll use SARIMA(1,1,1)(1,1,1,12)
print("\nFitting SARIMA(1,1,1)(1,1,1,12) model...")
model = SARIMAX(data['Passengers'],
                order=(1, 1, 1),
                seasonal_order=(1, 1, 1, 12),
                enforce_stationarity=False,
                enforce_invertibility=False)

results = model.fit()
print("Model fitted.")

# === Step 4: Print model summary ===
print("\nSARIMA Model Summary:")
print(results.summary())

# === Step 5: Forecast next 12 months ===
forecast = results.forecast(steps=12)
print("\nForecast for Next 12 Months:")
print(forecast)

# Create future index for plotting
future_index = pd.date_range(start=data.index[-1] + pd.DateOffset(months=1), periods=12, freq='MS')

# Plot original + forecast
plt.figure(figsize=(10, 5))
plt.plot(data['Passengers'], label='Original Data')
plt.plot(future_index, forecast, label='SARIMA Forecast', color='red')
plt.title("SARIMA Forecast for Airline Passengers")
plt.xlabel("Date")
plt.ylabel("Passengers")
plt.legend()
plt.grid(True)
plt.show()
