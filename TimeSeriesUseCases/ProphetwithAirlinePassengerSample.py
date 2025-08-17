
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

# === Step 1: Load data from local file ===
file_path = r"./DataSets/airline-passengers.csv"
data = pd.read_csv(file_path)

# === Step 2: Rename columns for Prophet ===
# Prophet expects 'ds' for date, 'y' for value
data.rename(columns={'Month': 'ds', 'Passengers': 'y'}, inplace=True)

# Convert 'ds' column to datetime
data['ds'] = pd.to_datetime(data['ds'])

print("Sample Data for Prophet:")
print(data.head())

# === Step 3: Create and fit the Prophet model ===
model = Prophet()
model.fit(data)

# === Step 4: Create future dates for prediction ===
future = model.make_future_dataframe(periods=12, freq='MS')  # 12 months into future

# === Step 5: Forecast using the model ===
forecast = model.predict(future)

# === Step 6: Print forecasted values ===
print("\nForecasted Values:")
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(12))

# === Step 7: Plot the forecast ===
fig1 = model.plot(forecast)
plt.title("Prophet Forecast: Airline Passengers")
plt.xlabel("Date")
plt.ylabel("Passengers")
plt.grid(True)
plt.show()

# === Step 8: Optional: Plot trend and seasonality separately ===
fig2 = model.plot_components(forecast)
plt.show()
