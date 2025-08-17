
# Import the libraries
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Initialize app
app = FastAPI(title="Warehouse Rack Location Prediction API")

# Load the dataset
df = pd.read_excel("./DataSets/warehouse_item_placement_sample.xlsx")
df['item_category'] = df['item_category'].astype(str).str.strip()
df['rack_id'] = df['rack_id'].astype(str).str.strip()

# Encode the rack_id column
le_rack = LabelEncoder()
df['rack_label'] = le_rack.fit_transform(df['rack_id'])

# Feature selection
features = ['item_category', 'picking_frequency', 'travel_time_to_rack', 'stored_quantity']
df = pd.get_dummies(df, columns=['item_category'])  # One-hot encode item_category

X = df.drop(columns=['rack_id', 'rack_label', 'item_code'])
y = df['rack_label']

# Split the model into train and test
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Define Pydantic models for request schema
class RackInput(BaseModel):
    item_category: str
    picking_frequency: int
    travel_time_to_rack: int
    stored_quantity: int

class RackPredictionRequest(BaseModel):
    items: List[RackInput]

# Define the prediction endpoint
@app.post("/predict-best-rack")
def predict_best_rack(data: RackPredictionRequest):
    # Convert incoming data to DataFrame using model_dump()
    input_df = pd.DataFrame([item.model_dump() for item in data.items])
    
    # One-hot encode item_category and align with training features
    input_df = pd.get_dummies(input_df)
    input_df = input_df.reindex(columns=X.columns, fill_value=0)

    # Predict the best rack labels
    preds = model.predict(input_df)
    best_racks = le_rack.inverse_transform(preds)

    # Prepare response
    results = []
    for item, rack in zip(data.items, best_racks):
        results.append({
            "item_category": item.item_category,
            "predicted_best_rack": rack
        })

    return {"predictions": results}
