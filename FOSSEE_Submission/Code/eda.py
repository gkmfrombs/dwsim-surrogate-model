import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# --- 1. Setup and Data Loading ---
data_dir = r"C:\Users\guddu\OneDrive\Desktop\Doc\dwsim-surrogate-model\data"
raw_csv = os.path.join(data_dir, "results.csv")

print(f"Loading data from {raw_csv}...")
df_raw = pd.read_csv(raw_csv)

# Separate converged and failed runs
df = df_raw[df_raw['Converged'] == True].copy()
df_failed = df_raw[df_raw['Converged'] == False].copy()

print(f"Total Runs: {len(df_raw)}")
print(f"Converged: {len(df)}")
print(f"Failed: {len(df_failed)}")

# Set plotting style for a clean, modern look
sns.set_theme(style="whitegrid", palette="muted")

# --- 2. Create the Dashboard ---
fig = plt.figure(figsize=(18, 12))
fig.suptitle("FOSSEE Distillation Surrogate - Exploratory Data Analysis", fontsize=18, fontweight='bold')

# Subplot 1: Convergence Rate (Pie Chart)
ax1 = plt.subplot(2, 3, 1)
ax1.pie([len(df), len(df_failed)], labels=['Converged', 'Failed'], autopct='%1.1f%%', 
        colors=['#2ecc71', '#e74c3c'], startangle=90, explode=(0.05, 0))
ax1.set_title("Simulation Convergence Rate")

# Subplot 2: Mass Balance Discrepancy (Scatter)
ax2 = plt.subplot(2, 3, 2)
# Re-calculate mass balance for visualization
F = 100.0
Benzene_IN = F * df['Benzene_Frac']
Benzene_OUT = ((F - df['Bottoms_Rate']) * df['x_D_Benzene']) + (df['Bottoms_Rate'] * df['x_B_Benzene'])

sns.scatterplot(x=Benzene_IN, y=Benzene_OUT, alpha=0.5, ax=ax2, color="#3498db")
# Plot the "Perfect Physics" line (y=x)
ax2.plot([20, 80], [20, 80], 'k--', lw=2, label="Perfect Balance (In = Out)")
ax2.set_title("Mass Balance Check")
ax2.set_xlabel("Total Benzene IN (mol/s)")
ax2.set_ylabel("Total Benzene OUT (mol/s)")
ax2.legend()

# Subplot 3: Feature Correlation Heatmap
ax3 = plt.subplot(2, 3, 3)
# Select key numerical columns for correlation
corr_cols = ['Feed_Temp_K', 'Feed_Press_Pa', 'Benzene_Frac', 'Reflux_Ratio', 
             'Bottoms_Rate', 'x_D_Benzene', 'x_B_Benzene', 'Q_C_kW']
corr_matrix = df[corr_cols].corr()

sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=False, 
            square=True, ax=ax3, annot_kws={"size": 8})
ax3.set_title("Feature Correlation Heatmap")

# Subplot 4: Input Distribution (Temperature vs Pressure)
ax4 = plt.subplot(2, 3, 4)
sns.histplot(data=df_raw, x="Feed_Temp_K", y="Feed_Press_Pa", bins=30, 
             pthresh=.1, cmap="mako", ax=ax4)
ax4.set_title("LHS Sampling Distribution (Temp vs Press)")

# Subplot 5: Distillate Purity Distribution
ax5 = plt.subplot(2, 3, 5)
sns.histplot(df['x_D_Benzene'], bins=40, kde=True, color="#9b59b6", ax=ax5)
ax5.set_title("Top Product Purity (x_D_Benzene)")
ax5.set_xlabel("Mole Fraction")

# Subplot 6: Energy Duty Distribution
ax6 = plt.subplot(2, 3, 6)
sns.histplot(df['Q_C_kW'], bins=40, kde=True, color="#e67e22", ax=ax6)
ax6.set_title("Condenser Duty Distribution (Q_C)")
ax6.set_xlabel("Energy (kW)")

# --- 3. Render and Save ---
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
eda_output_path = os.path.join(data_dir, "eda_dashboard.png")
plt.savefig(eda_output_path, dpi=300)
print(f"EDA Dashboard successfully saved to {eda_output_path}")

plt.show()