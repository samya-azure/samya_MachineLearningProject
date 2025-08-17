
# classification_model.py file

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# load the datasets

iris = load_iris()
x = iris.data
y = iris.target

# split into train and test
x_train,x_test,y_train,y_test =  train_test_split(x, y, test_size = 0.2, random_state = 42)


# train the model
model = RandomForestClassifier()
model.fit(x_train,y_train)

# Prediction
y_pred = model.predict(x_test)

# evaluation
accuracy = accuracy_score(y_test,y_pred)
print("Model Accuracy", accuracy)




