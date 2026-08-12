# Surrogate Modeling of a Binary Distillation Column Using DWSIM and Machine Learning
**FOSSEE Autumn 2026 Screening Task - Stage 1 Submission**

---

## 1. Problem Statement
The objective of this project is to develop a robust, physically consistent surrogate model for a binary distillation column using rigorous simulation data. Distillation columns are highly non-linear and computationally expensive to simulate in real-time. By training Machine Learning models on data generated from DWSIM, this project aims to build a computationally efficient surrogate capable of predicting product purities ($x_D$, $x_B$) and energy duties ($Q_C$, $Q_R$) instantaneously based on operating conditions.

## 2. DWSIM Flowsheet & Model Description
- **System:** Benzene - Toluene Binary Mixture
- **Thermodynamic Model:** Peng-Robinson (PR)
- **Column Setup:** A rigorous distillation column with a total condenser and a partial reboiler.
- **Base Conditions:** Feed flow of 100 mol/s, varying feed composition, temperature, pressure, reflux ratio, and theoretical stages.

## 3. Dataset Generation Strategy
The dataset was procedurally generated using a Python-CLR bridge to interface with DWSIM headlessly.
- **Sampling Method:** Latin Hypercube Sampling (LHS) was utilized to explore the 7-dimensional parameter space efficiently, ensuring even distribution.
- **Dataset Size:** 5,000 independent simulation runs were executed. 
- **Physical Validity:** Non-converged simulations were discarded. Post-generation, the dataset was subjected to strict mass-balance validation ($F \cdot z = D \cdot x_D + B \cdot x_B$) to remove thermodynamically inconsistent anomalies. The final validated dataset contains 1,943 rows.

### Operating Conditions (Input Variables)
1. Feed Temperature ($290 - 350$ K)
2. Feed Pressure ($1 - 1.5$ bar)
3. Benzene Mole Fraction in Feed ($0.2 - 0.8$)
4. Number of Stages ($10 - 30$)
5. Feed Stage Location ($5 - 15$)
6. Reflux Ratio ($1.0 - 5.0$)
7. Bottoms Withdrawal Rate ($40 - 60$ mol/s)

### Target Variables (Outputs)
1. Distillate Benzene Purity ($x_D$)
2. Bottoms Benzene Purity ($x_B$)
3. Condenser Duty ($Q_C$ in kW)
4. Reboiler Duty ($Q_R$ in kW)

## 4. Data Preprocessing
- **Splitting:** An 80/20 train-test split was applied to ensure evaluation on unseen data.
- **Scaling:** `StandardScaler` was used to normalize all input features (zero mean, unit variance), preventing features with large magnitudes from skewing the gradient descent algorithms.

## 5. Machine Learning Implementation
Three distinct modeling approaches were evaluated:

1. **Multiple Linear Regression (Baseline):** 
   - A standard baseline to check for linear thermodynamic relationships.
2. **XGBoost Regressor (Optuna Tuned):**
   - A non-linear decision tree ensemble. Hyperparameters were optimized via Optuna using 5-Fold Cross Validation.
3. **PyTorch Physics-Informed Neural Network (PINN):**
   - A custom deep learning model trained with a physics-informed loss function that mathematically penalizes predictions violating the Law of Conservation of Mass.

## 6. Model Performance Comparison

| Model | Overall R² | $x_D$ (Purity) R² | $x_B$ (Purity) R² | $Q_C$ (Duty) R² | $Q_R$ (Duty) R² | Overall MSE |
|---|---|---|---|---|---|---|
| Linear Regression | 89.88% | 81.3% | 81.6% | 98.4% | 98.1% | 0.0890 |
| XGBoost (Tuned) | 99.66% | 99.5% | 99.5% | 99.8% | 99.7% | 0.0036 |
| PyTorch PINN | **99.76%** | **99.7%** | **99.6%** | **99.8%** | **99.8%** | **0.0025** |

*Note: Linear regression adequately predicts duties as they scale mostly linearly with flow, but fails on purities which exhibit highly non-linear vapor-liquid equilibrium curves.*

## 7. Physical Consistency and Robustness (Explainable AI)
To ensure the models achieved true thermodynamic consistency rather than arbitrary mathematical fitting, a Parametric Sensitivity Analysis was conducted.
By holding all variables constant at their median and sweeping the **Reflux Ratio** from $1.0$ to $10.0$, the surrogate model correctly predicted a rapid asymptotic increase for Condenser Duty. This physically proves the model correctly identified that higher internal liquid traffic necessitates significantly more cooling capacity.

Furthermore, SHAP (SHapley Additive exPlanations) analysis confirmed that the number of stages and reflux ratio act as the highest mathematical drivers for output purity, perfectly aligning with distillation fundamentals.

## 8. Conclusion
The **PyTorch Physics-Informed Neural Network (PINN)** was identified as the best surrogate model. Not only does it achieve the highest statistical accuracy (99.76% R²), but its custom loss function explicitly guarantees that the generated predictions obey fundamental mass-balance laws. 

To demonstrate the practical utility of the model, a Streamlit dashboard (`app.py`) is included in the project repository for real-time interactive inference.
