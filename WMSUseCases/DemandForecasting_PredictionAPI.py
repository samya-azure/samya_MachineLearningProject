
# import the libraries
from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


app = FastAPI(title="Customer Order Prediction API")

# Load the dataset
df = pd.read_excel("./DataSets/customer_item_orders_sample.xlsx")
df['item_code'] = df['item_code'].astype(str).str.strip()
df['customer_code'] = df['customer_code'].astype(str).str.strip()

# set the Features and target
features = ['days_since_last_order', 'total_orders_last_30_days',
            'avg_order_interval', 'recently_viewed']
target = 'ordered_in_next_7_days'

# Split the model in train & test
X = df[features]
y = df[target]
#X_train, _, y_train, _ = train_test_split(X, y, test_size=0.3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the model
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Define the API Request body structure
class PredictionRequest(BaseModel):
    item_codes: List[str]
    customer_codes: List[str]

# Prediction API endpoint
@app.post("/predict")
def predict_order(data: PredictionRequest):
    # Filter matching rows with the dataset customer_item_orders_sample.xlsx
    selected = df[
        df['item_code'].isin(data.item_codes) &
        df['customer_code'].isin(data.customer_codes)
    ].copy()

    if selected.empty:
        return {"message": "No matching customer-item pairs found."}

    # Making Prediction
    X_selected = selected[features]
    preds = clf.predict(X_selected)
    labels = ['Will Order in next 7 days' if p == 1 else 'Will Not Order in next 7 days' for p in preds]

    # Building the Response
    results = []
    for cust, item, label in zip(selected['customer_code'], selected['item_code'], labels):
        results.append({
            "customer_code": cust,
            "item_code": item,
            "prediction": label
        })

    # return the result
    return {"predictions": results}


