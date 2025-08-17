
# import the libraries
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load and prepare data once during startup
df = pd.read_excel("./DataSets/warehouse_items_sample.xlsx")
df['item_code'] = df['item_code'].astype(str).str.strip()

# set the threshold limit 20, i.e, item move more than 20 times in last 30 days
threshold = 20
df['frequency_label'] = df['total_moves_last_30_days'].apply(
    lambda x: 'frequent' if x > threshold else 'infrequent'
)

feature_cols = ['length_cm', 'width_cm', 'height_cm', 'weight_kg',
                'total_moves_last_30_days', 'avg_daily_moves', 'last_moved_days_ago']
X = df[feature_cols]
y = df['frequency_label'].map({'infrequent': 0, 'frequent': 1})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# FastAPI app
app = FastAPI()

# Request model
class ItemCodes(BaseModel):
    codes: List[str]

# API endpoint
@app.post("/predict")
def predict_items_frequency(item_data: ItemCodes):
    # Strip & ensure strings
    input_codes = [str(code).strip() for code in item_data.codes]

    selected_items = df[df['item_code'].isin(input_codes)].copy()
    if selected_items.empty:
        raise HTTPException(status_code=404, detail="No matching item codes found in Excel")

    X_selected = selected_items[feature_cols]
    predictions = clf.predict(X_selected)
    prediction_labels = ['frequent' if p == 1 else 'infrequent' for p in predictions]

    results = []
    for code, label in zip(selected_items['item_code'], prediction_labels):
        results.append({"item_code": code, "predicted_frequency": label})

    return {"results": results}
