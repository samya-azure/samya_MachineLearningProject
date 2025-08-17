
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss
import warnings
warnings.filterwarnings("ignore")

# Function to plot rolling statistics
def plot_rolling_stats(series, window=12):
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()

    plt.figure(figsize=(12, 6))
    plt.plot(series, color='blue', label='Original')
    plt.plot(rolling_mean, color='red', label='Rolling Mean')
    plt.plot(rolling_std, color='green', label='Rolling Std Dev')
    plt.legend()
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()

# Function to perform ADF Test
def adf_test(series):
    print("ADF (Augmented Dickey-Fuller) Test:")
    result = adfuller(series.dropna())
    print(f"ADF Statistic  : {result[0]:.4f}")
    print(f"p-value        : {result[1]:.4f}")
    if result[1] < 0.05:
        print("The series is likely STATIONARY.")
    else:
        print("The series is likely NON-STATIONARY.")
    print()

# Function to perform KPSS Test
def kpss_test(series):
    print("KPSS (Kwiatkowski-Phillips-Schmidt-Shin) Test:")
    result = kpss(series.dropna(), regression='c')
    print(f"KPSS Statistic : {result[0]:.4f}")
    print(f"p-value        : {result[1]:.4f}")
    if result[1] > 0.05:
        print("The series is likely STATIONARY.")
    else:
        print("The series is likely NON-STATIONARY.")
    print()

# Full Stationarity Check Function
def check_stationarity(series, window=12):
    print("Plotting Rolling Statistics...")
    plot_rolling_stats(series, window)

    print("Running Statistical Tests...\n")
    adf_test(series)
    kpss_test(series)

# -----------------------------------
# Example Usage: Simulated Data
# -----------------------------------

# Simulate stationary data
np.random.seed(42)
stationary_data = pd.Series(np.random.normal(loc=0.0, scale=1.0, size=100))

# Simulate non-stationary data (with trend)
trend = np.linspace(1, 100, 100)
non_stationary_data = pd.Series(trend + np.random.normal(0, 5, 100))

# Choose which series to check
#series_to_check = non_stationary_data  # to test non-stationary
series_to_check = stationary_data  # to test stationary


# Run stationarity check
check_stationarity(series_to_check)
