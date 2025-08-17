
import pandas as pd
import joblib

# Load the model
model = joblib.load("app/my_model_new.pkl")  # Or use full path like "C:/Users/yourname/path/to/my_model_new.pkl"

# Create a sample input data (must match training features!)
sample_data = pd.DataFrame([{
    'PassengerId': 2,
    'Pclass': 2,
    'Sex': 0,        # Make sure to encode if your model expects it
    'Age': 39.0,
    'SibSp': 0,
    'Parch': 0,
    'Fare': 150.55
}])

# Predict
prediction = model.predict(sample_data)
print("*********************************")
print("Prediction:", prediction)
