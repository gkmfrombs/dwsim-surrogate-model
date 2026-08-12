# DWSIM Surrogate Modeling Project (FOSSEE Submission)

This repository contains a full Machine Learning surrogate model of a Benzene-Toluene binary distillation column, developed using Python and DWSIM.

## Repository Structure & FOSSEE Deliverables
* `DWSIM_Flowsheet.dwxmz` - The rigorous DWSIM simulation file.
* `Dataset.csv` - The 1,943 valid simulation runs used for training.
* `Report.md` (or `.pdf`) - The comprehensive methodology and analysis report.
* `Results_Summary.md` - A simple breakdown of the ML metrics and findings.
* `Code/` - The Python source files used for training and testing the models.
* `app.py` - An interactive Streamlit Web UI for real-time model inference.

## How to Run the Code

### 1. Environment Setup
You will need Python installed. To ensure all packages match the development environment, install the dependencies:
```bash
pip install pandas numpy scikit-learn xgboost optuna torch shap matplotlib streamlit
```

### 2. Opening the DWSIM Flowsheet
1. Open DWSIM (v8.6+ recommended).
2. Click File -> Open File.
3. Select `DWSIM_Flowsheet.dwxmz`.
4. Press `F5` to run the flowsheet manually.

### 3. Training the Models (Optional)
The pre-trained models are already provided in `data/models/`. If you wish to retrain them from scratch:
1. Open a terminal in the root directory.
2. Run the Linear Regression Baseline:
   ```bash
   python Code/train_lr.py
   ```
3. Run the XGBoost Tuner (Warning: Takes ~1 min):
   ```bash
   python Code/train_xgb.py
   ```
4. Run the PyTorch Physics-Informed Neural Network:
   ```bash
   python Code/train_ann.py
   ```

### 4. Explainable AI and Physics Verification
To prove physical consistency, you can run the explanation script which generates SHAP feature importance charts and Parametric Sensitivity trend lines:
```bash
python Code/explain.py
```
The output plots will be saved to `data/plots/`.

### 5. Running the Interactive Streamlit Web UI (Highly Recommended)
To interact with the final Surrogate Model in real-time using UI sliders, run the Streamlit app:
```bash
streamlit run app.py
```
This will automatically open a local web server in your browser.

## Assumptions Made
- Feed flow rate is assumed constant at 100 mol/s. 
- Distillation column operates with a total condenser and partial reboiler.
- Thermodynamic property package is Peng-Robinson (PR).
- Data samples that failed to converge in DWSIM or violated the Law of Conservation of Mass were filtered out during preprocessing.
