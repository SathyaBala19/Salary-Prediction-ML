import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

DATA_PATH = "data/Salary_Data.csv"
MODEL_PATH = "models/salary_model.pkl"
PLOT_PATH = "models/regression_plot.png"

def load_data(file_path: str) -> pd.DataFrame:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at path: {file_path}")
    
    data = pd.read_csv(file_path)

    return data

def validate_data(data: pd.DataFrame) -> None:
    required_columns = ["YearsExperienced", "Salary"]

    for column in required_columns:
        if column not in data.columns:
            raise ValueError(f"Missing required column: {column}")
        
    if data[required_columns].isnull().sum().sum() > 0:
        raise TypeError("Dataset contains missing values.")
    
    if not np.issubtype(data["YearsExperience"].dtype, np.number):
        raise TypeError("YearsExperience column must be numeric.")
    
    if not np.issubtype(data["Salary"].dtype, np.number):
        raise TypeError("Salary column must be numeric.")
    
    if (data["YearsExperience"] < 0).any():
        raise ValueError("YearsExperience cannot contain negative values.")
    
    if (data["Salary"] <= 0).any():
        raise ValueError("Salary must be greater than zero.")
    
def prepare_features(data: pd.DataFrame):
    X = data[["YearsExperience"]]
    y = data[["Salary"]]

    return X, y

def build_model() -> Pipeline:
    model_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("regressor", LinearRegression())
        ]
    )

    return model_pipeline

def evaluate_model(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    metrics = {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    }

    return metrics

def perform_cross_validation(model, X, y) -> float:
    scores = cross_val_score(
        model,
        X,
        y,
        cv = 5,
        scoring="r2"
    )

    return scores.mean()

def save_model(model, model_path: str) -> None:
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)

def plot_regression_line(model, X, y, plot_path: str) -> None:
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)

    X_sorted = X.sort_values(by="YearsExperience")
    y_pred_sorted = model.predict(X_sorted)

    plt.figure(figsize=(8, 5))
    plt.scatter(X, y, label="Actual Data")
    plt.plot(X_sorted, y_pred_sorted, label="Regression Line")
    plt.xlabel("Years of Experience")
    plt.ylabel("Salary")
    plt.title("Salary Prediction using Linear Regression")
    plt.legend()
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

def train_model() -> None:
    print("Loading dataset...")
    data = load_data(DATA_PATH)

    print("Validating dataset...")
    validate_data(data)

    print("Preparing features...")
    X, y = prepare_features(data)

    print("Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Building model pipeline...")
    model = build_model()

    print("Training Model...")
    model.fit(X_train, y_train)

    print("Evaluating model...")
    metrics = evaluate_model(model, X_test, y_test)

    print("\nModel Evaluation Metrics")
    print("---------------------------")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.2f}")

    print("\nPerforming cross-validation...")
    cv_score  =perform_cross_validation(model, X, y)
    print(f"Average Cross Validation R2 Score: {cv_score:.2f}")

    print("\nSaving Model...")
    save_model(model, MODEL_PATH)

    print("Saving regression plot...")
    plot_regression_line(model, X, y, PLOT_PATH)

    regressor = model.named_steps["regressor"]

    print("\nModel Parameters")
    print("--------------------")
    print(f"Coefficient: {regressor.coef_[0]:.2f}")
    print(f"Intercept: {regressor.intercept_:.2f}")

    print("\nTraining completed successfully.")

if __name__ == "__main__":
    train_model()