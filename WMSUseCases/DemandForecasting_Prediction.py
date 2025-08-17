
# import the libraries
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load the dataset of sample excel
df = pd.read_excel("./DataSets/customer_item_orders_sample.xlsx")  # update path if needed

# Ensure item_code and customer_code are strings
df['item_code'] = df['item_code'].astype(str).str.strip()
df['customer_code'] = df['customer_code'].astype(str).str.strip()

# Define features and label
features = ['days_since_last_order', 'total_orders_last_30_days', 
            'avg_order_interval', 'recently_viewed']
target = 'ordered_in_next_7_days'

# Prepare X and y
X = df[features]
y = df[target]

# Split the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the model
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Prediction on test set
y_pred = clf.predict(X_test)

# Evaluation
print(classification_report(y_test, y_pred))

# --- Predict for selected customers/items ---

# Example prediction for specific customer-item pairs
selected = df[df['item_code'].isin(['P001', 'P004'])].copy()  # adjust item codes

if selected.empty:
    print("No matching items found.")
else:
    X_selected = selected[features]
    preds = clf.predict(X_selected)
    labels = ['Will Order' if p == 1 else 'Will Not Order' for p in preds]

    for cust, item, label in zip(selected['customer_code'], selected['item_code'], labels):
        print(f"Customer {cust} is predicted to '{label}' item {item} in next 7 days.")
