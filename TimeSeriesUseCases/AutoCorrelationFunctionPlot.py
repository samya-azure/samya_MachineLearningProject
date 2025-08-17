
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf, adfuller
import warnings
warnings.filterwarnings("ignore")

# Load dataset from local CSV
file_path = r"./DataSets/airline-passengers.csv"
data = pd.read_csv(file_path)

# Convert Month to datetime and set index
data['Month'] = pd.to_datetime(data['Month'])
data.set_index('Month', inplace=True)
passengers = data['Passengers']

# Plot original time series
plt.figure(figsize=(10, 4))
plt.plot(passengers, label="Monthly Airline Passengers")
plt.title("Airline Passengers Over Time")
plt.xlabel("Year")
plt.ylabel("Number of Passengers")
plt.grid(True)
plt.legend()
plt.show()

# Plot rolling mean and standard deviation
rolling_mean = passengers.rolling(window=12).mean()
rolling_std = passengers.rolling(window=12).std()

plt.figure(figsize=(10, 4))
plt.plot(passengers, label='Original')
plt.plot(rolling_mean, label='Rolling Mean (12 months)', color='red')
plt.plot(rolling_std, label='Rolling Std (12 months)', color='green')
plt.title("Rolling Mean & Std Deviation")
plt.legend()
plt.grid(True)
plt.show()

# Perform Augmented Dickey-Fuller Test
result = adfuller(passengers)
print("ADF Test Results:")
print(f"ADF Statistic : {result[0]:.4f}")
print(f"p-value       : {result[1]:.4f}")
print(f"# Lags Used   : {result[2]}")
print(f"# Observations: {result[3]}")

# Conclusion
if result[1] < 0.05:
    print("\nConclusion: The series is likely **stationary** (p < 0.05)")
else:
    print("\nConclusion: The series is likely **not stationary** (p ≥ 0.05)")

# Optional: Print ACF values
acf_values = acf(passengers, nlags=20)
print("\nACF Values (up to Lag 20):")
for i, val in enumerate(acf_values):
    print(f"Lag {i}: {val:.4f}")

# Plot ACF
plot_acf(passengers, lags=20)
plt.title("ACF Plot: Airline Passengers")
plt.xlabel("Lag")
plt.ylabel("Autocorrelation")
plt.grid(True)
plt.show()
