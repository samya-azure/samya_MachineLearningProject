
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
warnings.filterwarnings("ignore")

# Load dataset from local file
file_path = r"./DataSets/airline-passengers.csv"
data = pd.read_csv(file_path)

# Parse dates and set index
data['Month'] = pd.to_datetime(data['Month'])
data.set_index('Month', inplace=True)

# Target series: number of passengers
passengers = data['Passengers']

# Decompose the series (additive model)
decomposition = seasonal_decompose(passengers, model='additive', period=12)

# Extract trend and seasonality components
trend = decomposition.trend
seasonal = decomposition.seasonal

# Plot trend and seasonality
plt.figure(figsize=(12, 6))

# Plot trend
plt.subplot(2, 1, 1)
plt.plot(trend, label='Trend', color='blue')
plt.title("Trend Component")
plt.grid(True)
plt.legend()

# Plot seasonality
plt.subplot(2, 1, 2)
plt.plot(seasonal, label='Seasonality', color='green')
plt.title("Seasonal Component")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
