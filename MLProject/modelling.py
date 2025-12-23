import pandas as pd
import mlflow
import mlflow.sklearn
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


data_path = "student_performance_clean.csv"
df = pd.read_csv(data_path)

X = df.drop("FinalGrade", axis=1)
y = df["FinalGrade"]

# traint tes split 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# mlflow
mlflow.set_experiment("Student_Performance_Experiment")
with mlflow.start_run():

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Metric
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("R2", r2)

    # Log model
    mlflow.sklearn.log_model(model, "student_model")

    print("Training selesai")
    print(f"MAE  : {mae}")
    print(f"RMSE : {rmse}")
    print(f"R2   : {r2}")
