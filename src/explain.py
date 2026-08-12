import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib
import os

# --- 1. Load Everything ---
print("Loading Models and Data...")
models_dir = r"C:\Users\guddu\OneDrive\Desktop\Doc\dwsim-surrogate-model\data\models"
data_dir = r"C:\Users\guddu\OneDrive\Desktop\Doc\dwsim-surrogate-model\data\processed"
plots_dir = r"C:\Users\guddu\OneDrive\Desktop\Doc\dwsim-surrogate-model\data\plots"
os.makedirs(plots_dir, exist_ok=True)

# Load the XGBoost MultiOutputRegressor model
model = joblib.load(os.path.join(models_dir, "xgb_model.pkl"))
scaler_X = joblib.load(os.path.join(models_dir, "scaler_X.pkl"))
scaler_y = joblib.load(os.path.join(models_dir, "scaler_y.pkl"))

X_train_scaled = pd.read_csv(os.path.join(data_dir, "X_train.csv"))
features = X_train_scaled.columns.tolist()
targets = ["x_D_Benzene", "x_B_Benzene", "Q_C_kW", "Q_R_kW"]

# --- 2. SHAP Analysis (Explainable AI) ---
print("Generating SHAP Summary Plot for Condenser Duty (Q_C)...")
# Since we used MultiOutputRegressor, it trained 4 separate trees.
# We want to explain Q_C_kW, which is index 2.
qc_model = model.estimators_[2]

# Create a TreeExplainer
explainer = shap.TreeExplainer(qc_model)
# Calculate SHAP values for a random sample of 500 rows to keep it fast
shap_values = explainer.shap_values(X_train_scaled.sample(500, random_state=42))

# Generate the plot
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_train_scaled.sample(500, random_state=42), feature_names=features, show=False)
plt.title("SHAP Summary Plot: What Drives Condenser Duty?")
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "shap_summary_Q_C.png"), dpi=300, bbox_inches='tight')
plt.close()
print("Saved SHAP plot to data/plots/shap_summary_Q_C.png")

# --- 3. Parametric Sensitivity Analysis ---
print("Running Parametric Sensitivity Analysis (Reflux Ratio vs Condenser Duty)...")

# We want to hold 6 variables perfectly constant at their median, while changing Reflux Ratio.
# Let's get the median of the UNSCALED training data to build our "Base Case"
unscaled_X_train = pd.DataFrame(scaler_X.inverse_transform(X_train_scaled), columns=features)
base_case = unscaled_X_train.median().values

# Create 100 fake scenarios where ONLY Reflux Ratio changes
# Reflux Ratio is feature index 5. Let's sweep it from 1.0 to 10.0
reflux_sweep = np.linspace(1.0, 10.0, 100)

scenarios = []
for rr in reflux_sweep:
    scenario = base_case.copy()
    scenario[5] = rr  # Force the reflux ratio
    scenarios.append(scenario)

# Convert to DataFrame and Scale it so the AI can read it
scenarios_df = pd.DataFrame(scenarios, columns=features)
scenarios_scaled = scaler_X.transform(scenarios_df)

# Ask the AI to predict the targets
predictions_scaled = model.predict(scenarios_scaled)

# Unscale the predictions so we can read the raw kW values
predictions_unscaled = scaler_y.inverse_transform(predictions_scaled)
predicted_QC = predictions_unscaled[:, 2] # Extract Q_C (index 2)

# Plot the physics curve
plt.figure(figsize=(8, 5))
plt.plot(reflux_sweep, predicted_QC, color='red', linewidth=2.5)
plt.title("AI Physics Verification: Reflux Ratio vs Condenser Duty")
plt.xlabel("Reflux Ratio")
plt.ylabel("Predicted Condenser Duty (kW)")
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "sensitivity_reflux_QC.png"), dpi=300)
plt.close()

print("Saved Sensitivity plot to data/plots/sensitivity_reflux_QC.png")
print("\nPhase 7 Complete! The AI has successfully explained its logic.")
