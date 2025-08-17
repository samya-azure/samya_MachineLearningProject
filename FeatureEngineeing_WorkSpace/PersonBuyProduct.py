
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

# Step 1: Create the raw data
data = {
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 22],
    'Country': ['India', 'USA', 'UK'],
    'JoinDate': ['2024-05-10', '2023-12-01', '2024-01-15'],
    'Purchased': ['Yes', 'No', 'Yes']
}

df = pd.DataFrame(data)

# Step 2: Drop unnecessary column
df = df.drop('Name', axis=1)  # Name isn't useful for prediction

# Step 3: Convert categorical 'Country' using one-hot encoding
df = pd.get_dummies(df, columns=['Country'])

# Step 4: Convert 'JoinDate' to datetime and extract features
df['JoinDate'] = pd.to_datetime(df['JoinDate'])
df['JoinMonth'] = df['JoinDate'].dt.month
df['JoinDayOfWeek'] = df['JoinDate'].dt.dayofweek
df = df.drop('JoinDate', axis=1)  # Original column dropped

# Step 5: Convert 'Purchased' Yes/No to 1/0
df['Purchased'] = df['Purchased'].map({'Yes': 1, 'No': 0})

# Final Data
print(df)
