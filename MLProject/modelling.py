import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# set experiment
mlflow.set_experiment("Student_Performance_Experiment")

df = pd.read_csv("student_performance_clean.csv")

X = df.drop("FinalGrade", axis=1)
y = df["FinalGrade"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# Loggiug
mlflow.log_metric("MAE", mae)
mlflow.log_metric("RMSE", rmse)
mlflow.log_metric("R2", r2)

mlflow.sklearn.log_model(model, "student_model")

print("Training selesai")
print("MAE :", mae)
print("RMSE:", rmse)
print("R2  :", r2)
