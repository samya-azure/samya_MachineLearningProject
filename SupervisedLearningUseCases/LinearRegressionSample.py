
from sklearn.linear_model import LinearRegression

# Example data: hours studied vs marks scored
x = [[1], [2], [3], [4], [5]]  # Hours
y = [50, 60, 65, 70, 75]  # Marks

# creating machine learning model using Linear Regression Algorithm
model = LinearRegression()

# Train the model, here .fit means learning of model
model.fit(x,y)

# predict the model for marks if study for the hour 6
#predicted = model.predict([[6]])   #----commented

# predict the model for marks if study for the hours 6,7 & 8
predicted = model.predict([[6], [7], [8]])

# print the result
print(predicted)



