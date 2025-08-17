
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# -------------------------------
# Step 1: Load Dataset
# -------------------------------
df = pd.read_csv("./DataSets/Kaggle_Dataset_LoanApplication.csv")

# -------------------------------
# Step 2: Preprocess Data
# -------------------------------

# Drop rows with missing values
df = df.dropna()

# Convert categorical columns to numeric using LabelEncoder
label_cols = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']
label_encoders = {}

for col in label_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Features and label
X = df[['Gender', 'Married', 'Education', 'Self_Employed',
        'ApplicantIncome', 'CoapplicantIncome',
        'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Property_Area']]

y = df['Loan_Status']  # 1 = approved, 0 = not approved

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# -------------------------------
# Step 3: Build Neural Network
# -------------------------------
model = Sequential()
model.add(Dense(16, input_dim=X.shape[1], activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # Binary output

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# -------------------------------
# Step 4: Train the Model
# -------------------------------
model.fit(X_train, y_train, epochs=50, batch_size=10, verbose=1)

# -------------------------------
# Step 5: Evaluate
# -------------------------------
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {accuracy:.2f}")

# -------------------------------
# Step 6: Predict New Customer
# -------------------------------

# Example new customer:
# Gender=Male (1), Married=Yes (1), Education=Graduate (1), Self_Employed=No (0)
# ApplicantIncome=6000, CoapplicantIncome=1500, LoanAmount=200, Loan_Amount_Term=360
# Credit_History=1 (good), Property_Area=Urban (encoded as 2)

new_customer = [[1, 1, 1, 0, 6000, 1500, 200, 360, 1, 2]]

# Scale using the same scaler used for training
new_customer_scaled = scaler.transform(new_customer)

# Predict
prediction = model.predict(new_customer_scaled)

# Output result
print(f"\nPrediction Probability: {prediction[0][0]:.3f}")
if prediction[0][0] > 0.5:
    print("Prediction: Loan Approved!")
else:
    print("Prediction: Loan Rejected.")
