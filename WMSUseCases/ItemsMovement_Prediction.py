

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load data
df = pd.read_excel("./DataSets/warehouse_items_sample.xlsx")

#  Convert item_code to string for safe comparison
df['item_code'] = df['item_code'].astype(str).str.strip()

# Create target label based on total_moves_last_30_days threshold
threshold = 20
df['frequency_label'] = df['total_moves_last_30_days'].apply(lambda x: 'frequent' if x > threshold else 'infrequent')

# Define feature columns
feature_cols = ['length_cm', 'width_cm', 'height_cm', 'weight_kg', 
                'total_moves_last_30_days', 'avg_daily_moves', 'last_moved_days_ago']
X = df[feature_cols]
y = df['frequency_label']
y_encoded = y.map({'infrequent': 0, 'frequent': 1})

# Split and train model
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Evaluate model
y_pred = clf.predict(X_test)
#print(classification_report(y_test, y_pred, target_names=['infrequent', 'frequent'], zero_division=0))

# Predict for two existing items using item_code
selected_item_codes = ['IT001', 'IT004']  # Change these as needed

# Select rows matching item codes
selected_items = df[df['item_code'].isin(selected_item_codes)].copy()

# Ensure we got matching rows
if selected_items.empty:
    print("No matching item codes found in Excel.")
else:
    X_selected = selected_items[feature_cols]
    selected_preds = clf.predict(X_selected)
    selected_labels = ['frequent' if p == 1 else 'infrequent' for p in selected_preds]

    # Show results
    for code, label in zip(selected_items['item_code'], selected_labels):
        print(f"Item code {code} is predicted as: {label}")
