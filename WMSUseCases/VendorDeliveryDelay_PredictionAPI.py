
# import the libraries
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Initialize app
app = FastAPI(title="Vendor Delivery Delay Prediction API")

# Load the dataset
df = pd.read_excel("./DataSets/vendor_delivery_sample.xlsx")
df['vendor_code'] = df['vendor_code'].astype(str).str.strip()
df['item_code'] = df['item_code'].astype(str).str.strip()

# set the Features and Target
features = ['shipping_distance_km', 'promised_days', 'prior_delays_count', 'weather_issue']
target = 'delay_flag'

# split the model into train and test
X = df[features]
y = df[target]
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.3, random_state=42)

# Train  the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Define the API Request body structure
class DelayPredictionRequest(BaseModel):
    vendor_codes: List[str]
    item_codes: List[str]

# Prediction API endpoint
@app.post("/predict-delay")
def predict_delay(data: DelayPredictionRequest):
    # Filter matching rows with the dataset vendor_delivery_sample.xlsx
    selected = df[
        df['vendor_code'].isin(data.vendor_codes) &
        df['item_code'].isin(data.item_codes)
    ].copy()

    if selected.empty:
        return {"message": "No matching vendor-item pairs found."}

    # Making Prediction
    X_selected = selected[features]
    preds = model.predict(X_selected)
    labels = ['Will Delay' if p == 1 else 'On Time' for p in preds]

    # Building the Response
    results = []
    for vendor, item, label in zip(selected['vendor_code'], selected['item_code'], labels):
        results.append({
            "vendor_code": vendor,
            "item_code": item,
            "prediction": label
        })

    # return the result
    return {"predictions": results}
