import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

def preprocess_data(input_csv, output_dir):
    """
    Loads raw DWSIM simulation data, cleans it using chemical engineering 
    first-principles (mass balance & thermodynamics), and prepares it for Machine Learning.
    """
    print(f"--- 1. Loading Raw Data ---")
    df = pd.read_csv(input_csv)
    initial_count = len(df)
    print(f"Initial dataset size: {initial_count} rows")

    print(f"\n--- 2. Convergence Filtering ---")
    # In DWSIM, if a simulation doesn't converge, the results are mathematically invalid.
    # We strictly filter for rows where 'Converged' is True.
    df = df[df["Converged"] == True].copy()
    converged_count = len(df)
    print(f"Dropped {initial_count - converged_count} non-converged rows.")

    print(f"\n--- 3. Strict Mass Balance Conservation ---")
    # The Law of Conservation of Mass must hold true for our data to be real physics.
    # Component Balance for Benzene: Feed_Benzene = Distillate_Benzene + Bottoms_Benzene
    # F * z = D * x_D + B * x_B
    
    # Calculate actual component flows
    F = df["F_mol_s"]
    z = df["Benzene_Frac"]
    D = df["D_mol_s"]
    B = df["Bottoms_Rate"] # The bottoms rate specification we set in DWSIM
    
    benzene_in = F * z
    benzene_out = (D * df["x_D_Benzene"]) + (B * df["x_B_Benzene"])
    
    # Calculate percentage error. If DWSIM did the math right, this should be very close to 0.
    mass_balance_error_percent = np.abs((benzene_out - benzene_in) / benzene_in) * 100
    
    # We enforce a strict 1% maximum error threshold.
    # Anything higher means the simulator converged on a "loose" tolerance or numerical glitch.
    valid_mass_balance = mass_balance_error_percent <= 1.0
    df = df[valid_mass_balance].copy()
    mb_count = len(df)
    print(f"Dropped {converged_count - mb_count} rows violating 1% mass balance tolerance.")

    print(f"\n--- 4. Thermodynamic Logic Checks ---")
    # For a distillation column separating a light key (Benzene) from a heavy key (Toluene):
    # The purity of Benzene at the top (x_D) MUST be greater than the feed (z).
    # The purity of Benzene at the bottom (x_B) MUST be less than the feed (z).
    # Logic: x_D > z > x_B
    
    valid_thermo = (df["x_D_Benzene"] > df["Benzene_Frac"]) & (df["Benzene_Frac"] > df["x_B_Benzene"])
    df = df[valid_thermo].copy()
    final_count = len(df)
    print(f"Dropped {mb_count - final_count} rows violating thermodynamic logic.")
    print(f"Final robust dataset size: {final_count} rows.")

    print(f"\n--- 5. Machine Learning Preparation ---")
    # We define our inputs (Features - X) and what we want to predict (Targets - y)
    
    # Features (7 Inputs)
    features = [
        "Feed_Temp_K", "Feed_Press_Pa", "Benzene_Frac", "Stages", 
        "Feed_Stage", "Reflux_Ratio", "Bottoms_Rate"
    ]
    X = df[features]
    
    # Targets (4 Outputs)
    targets = ["x_D_Benzene", "x_B_Benzene", "Q_C_kW", "Q_R_kW"]
    y = df[targets]
    
    # 80/20 Train-Test Split
    # We hide 20% of the data from the AI during training to test if it actually learned the physics.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Normalization (StandardScaler)
    # Neural Networks and Gradient Boosting work best when all inputs are on the same scale (mean=0, variance=1)
    # Note: We ONLY fit the scaler on the training data to prevent "data leakage" from the test set.
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)
    
    # Save the processed data for Phase 6 (Machine Learning)
    os.makedirs(output_dir, exist_ok=True)
    
    import joblib
    # Convert scaled numpy arrays back to Pandas DataFrames for easy saving
    pd.DataFrame(X_train_scaled, columns=features).to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
    pd.DataFrame(X_test_scaled, columns=features).to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
    pd.DataFrame(y_train_scaled, columns=targets).to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    pd.DataFrame(y_test_scaled, columns=targets).to_csv(os.path.join(output_dir, "y_test.csv"), index=False)
    
    # Save the scalers for Phase 8 (UI Deployment)
    models_dir = os.path.join(os.path.dirname(output_dir), "models")
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(scaler_X, os.path.join(models_dir, "scaler_X.pkl"))
    joblib.dump(scaler_y, os.path.join(models_dir, "scaler_y.pkl"))
    
    print(f"Data saved to {output_dir}")
    print(f"Scalers saved to {models_dir}")
    print("Preprocessing Complete. Ready for AI Modeling!")

if __name__ == "__main__":
    input_file = r"C:\Users\guddu\OneDrive\Desktop\Doc\dwsim-surrogate-model\data\results.csv"
    output_directory = r"C:\Users\guddu\OneDrive\Desktop\Doc\dwsim-surrogate-model\data\processed"
    preprocess_data(input_file, output_directory)
