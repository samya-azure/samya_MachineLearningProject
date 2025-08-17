
# Import libraries
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Initialize FastAPI app
app = FastAPI(title="Customer Engagement Channel Prediction API")

# Load the dataset
df = pd.read_excel("./DataSets/customer_engagement_channel_sample.xlsx")
df['customer_code'] = df['customer_code'].astype(str).str.strip()

# Define features and target column
features = ['email_response', 'sms_response', 'call_response', 'whatsapp_response', 'total_purchases']
target = 'preferred_channel'

# split the data into train and test
X = df[features]
y = df[target]
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Define request body schema
class EngagementPredictionRequest(BaseModel):
    customer_codes: List[str]

# Define prediction endpoint
@app.post("/predict-customer-engage-channel")
def predict_channel(data: EngagementPredictionRequest):
    # Filter rows for the given customer codes
    selected = df[df['customer_code'].isin(data.customer_codes)].copy()

    if selected.empty:
        return {"message": "No matching customers found."}

    # Make predictions
    X_selected = selected[features]
    preds = model.predict(X_selected)

    # Prepare response
    results = []
    for customer, channel in zip(selected['customer_code'], preds):
        results.append({
            "customer_code": customer,
            "recommended_channel": f"Should be engaged via {channel}"
        })

    return {"predictions": results}
