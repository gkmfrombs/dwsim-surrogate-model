import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import optuna
import joblib
import os

# Suppress Optuna logs to keep the terminal clean
optuna.logging.set_verbosity(optuna.logging.WARNING)

def load_data(data_dir):
    X_train = pd.read_csv(os.path.join(data_dir, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(data_dir, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv"))
    y_test = pd.read_csv(os.path.join(data_dir, "y_test.csv"))
    return X_train, X_test, y_train, y_test

def objective(trial, X, y):
    """
    Optuna objective function. 
    It tests different 'recipes' (hyperparameters) and scores them using 5-Fold Cross Validation.
    """
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 9),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'random_state': 42
    }
    
    # We use MultiOutputRegressor to train a separate XGBoost tree for each of our 4 targets
    base_model = xgb.XGBRegressor(**params)
    model = MultiOutputRegressor(base_model)
    
    # 5-Fold Cross Validation: 
    # We split our training data into 5 chunks. We train on 4 chunks and validate on 1. 
    # We repeat this 5 times and average the score. This proves our model isn't "overfitting".
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Calculate negative MSE (scikit-learn convention requires maximizing scores)
    scores = cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error')
    
    # Return the absolute Mean Squared Error
    return abs(scores.mean())

def train_xgboost():
    print("--- 1. Loading Preprocessed Data ---")
    data_dir = r"C:\Users\guddu\OneDrive\Desktop\Doc\dwsim-surrogate-model\data\processed"
    models_dir = r"C:\Users\guddu\OneDrive\Desktop\Doc\dwsim-surrogate-model\data\models"
    X_train, X_test, y_train, y_test = load_data(data_dir)

    print("--- 2. Starting Optuna Hyperparameter Tuning (with 5-Fold CV) ---")
    print("Searching for the perfect XGBoost configuration... (Testing 15 combinations)")
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=15)
    
    print("\nBest Hyperparameters Found:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    print("\n--- 3. Training Final XGBoost Model ---")
    best_params = study.best_params
    best_params['random_state'] = 42
    
    final_base_model = xgb.XGBRegressor(**best_params)
    final_model = MultiOutputRegressor(final_base_model)
    
    # Train the final model on the ENTIRE 80% training dataset
    final_model.fit(X_train, y_train)

    print("--- 4. Evaluating on the Hidden Test Set ---")
    y_pred = final_model.predict(X_test)
    
    overall_mse = mean_squared_error(y_test, y_pred)
    overall_r2 = r2_score(y_test, y_pred)
    
    print(f"Overall Mean Squared Error (MSE): {overall_mse:.4f}")
    print(f"Overall R-squared (R2) Score:     {overall_r2:.4f}")
    
    targets = ["x_D_Benzene", "x_B_Benzene", "Q_C_kW", "Q_R_kW"]
    print("\nMetrics per Target:")
    for i, target in enumerate(targets):
        target_mse = mean_squared_error(y_test.iloc[:, i], y_pred[:, i])
        target_r2 = r2_score(y_test.iloc[:, i], y_pred[:, i])
        print(f"  {target}: R2 = {target_r2:.4f}, MSE = {target_mse:.4f}")

    print("\n--- 5. Saving the Model ---")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "xgb_model.pkl")
    joblib.dump(final_model, model_path)
    print(f"Model saved successfully to {model_path}")

if __name__ == "__main__":
    train_xgboost()
