
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from mlflow.models.signature import infer_signature
import pandas as pd
import matplotlib.pyplot as plt

# Create synthetic data
X, y = make_regression(n_samples=100, n_features=1, noise=15, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Start MLflow run
with mlflow.start_run():

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    mlflow.log_param("model_type", "LinearRegression")
    mlflow.log_metric("mse", mse)

    # Add input example and signature
    signature = infer_signature(X_test, y_pred)
    mlflow.sklearn.log_model(
        sk_model=model,
        name="model",
        input_example=X_test[:5],
        signature=signature
    )

    # Plot and log the regression result
    plt.scatter(X_test, y_test, label="Actual")
    plt.plot(X_test, y_pred, color='red', label="Prediction")
    plt.legend()
    plt.title("Regression Plot")
    plt.savefig("regression_plot.png")
    mlflow.log_artifact("regression_plot.png")

    print("Run complete. Check MLflow UI.")
