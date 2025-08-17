
# import the libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# Load the Titanic.csv datasets
df = pd.read_csv("./DataSets/titanic.csv")

# select useful features and drop rows with missing data
df = df [["Survived", "Pclass", "Sex", "Age", "Fare"]].dropna()

# convert the category Sex into numeric value
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})


# Split data into input (X) and output (y)
X = df[["Pclass", "Sex", "Age", "Fare"]]
y = df["Survived"]

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# prediction of test data
y_pred = model.predict(X_test)

# evaluate the model
accuracy = accuracy_score(y_test,y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# prediction the survivol of a new passenger
# Example: Passenger Class: 2nd, Sex : female, age 30, fare $50

new_passenger = pd.DataFrame([[2, 1, 30, 50]], columns=["Pclass", "Sex", "Age", "Fare"])
prediction = model.predict(new_passenger)
result = "Survived" if prediction[0] == 1 else "Did NOT Survive"
print("\nPrediction for new passenger: ", result)


