import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

def train_baseline_model():
    print("--- 1. Loading Preprocessed Data ---")
    data_dir = r"C:\Users\guddu\OneDrive\Desktop\Doc\dwsim-surrogate-model\data\processed"
    models_dir = r"C:\Users\guddu\OneDrive\Desktop\Doc\dwsim-surrogate-model\data\models"
    
    # Load the normalized features and targets
    X_train = pd.read_csv(os.path.join(data_dir, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(data_dir, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv"))
    y_test = pd.read_csv(os.path.join(data_dir, "y_test.csv"))

    print("--- 2. Training Multiple Linear Regression Baseline ---")
    # A Linear Regression model assumes a straight-line relationship between inputs and outputs.
    # We train one model that predicts all 4 targets simultaneously.
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    
    print("--- 3. Evaluating the Model ---")
    # We ask the model to predict the answers for the test set (which it has never seen before)
    y_pred = lr_model.predict(X_test)
    
    # We compare its predictions (y_pred) to the actual true answers (y_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Overall Mean Squared Error (MSE): {mse:.4f}")
    print(f"Overall R-squared (R2) Score:     {r2:.4f}")
    
    # Let's break it down per target variable to see where it struggles
    targets = ["x_D_Benzene", "x_B_Benzene", "Q_C_kW", "Q_R_kW"]
    print("\nMetrics per Target:")
    for i, target in enumerate(targets):
        target_mse = mean_squared_error(y_test.iloc[:, i], y_pred[:, i])
        target_r2 = r2_score(y_test.iloc[:, i], y_pred[:, i])
        print(f"  {target}: R2 = {target_r2:.4f}, MSE = {target_mse:.4f}")

    print("\n--- 4. Saving the Model ---")
    # We save the trained brain to disk so we can use it in the web app later!
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "lr_baseline_model.pkl")
    joblib.dump(lr_model, model_path)
    print(f"Model saved successfully to {model_path}")

if __name__ == "__main__":
    train_baseline_model()
