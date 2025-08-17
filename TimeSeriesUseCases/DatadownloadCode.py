
import pandas as pd

# Download from URL
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
data = pd.read_csv(url)

# Save to local CSV
data.to_csv("./DataSets/airline-passengers.csv", index=False)
print("Dataset saved as 'airline-passengers.csv'")
