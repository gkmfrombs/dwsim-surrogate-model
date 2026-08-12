# DWSIM Surrogate Modeling Project (FOSSEE Submission)

**🚀 Live Interactive App:** [Click here to view the live Surrogate Model!](https://dwsim-surrogate-model-gkm-thx8crs2gqn4ubrlhj4lco.streamlit.app/)

This repository contains a full Machine Learning surrogate model of a Benzene-Toluene binary distillation column, developed using Python and DWSIM.

## Repository Structure & FOSSEE Deliverables
* `DWSIM_Flowsheet.dwxmz` - The rigorous DWSIM simulation file.
* `Dataset.csv` - The 1,943 valid simulation runs used for training.
* `Report.md` (or `.pdf`) - The comprehensive methodology and analysis report.
* `Results_Summary.md` - A brief breakdown of the ML metrics and findings.
* `Code/` - The Python source files used for data generation, preprocessing, and model training.
* `app.py` - An interactive Streamlit Web UI for real-time model inference.

## Execution Instructions

### 1. Environment Setup
To ensure all packages match the development environment, please install the dependencies:
```bash
pip install pandas numpy scikit-learn xgboost optuna torch shap matplotlib streamlit
```

### 2. Opening the DWSIM Flowsheet
1. Open DWSIM (v8.6+ recommended).
2. Click File -> Open File.
3. Select `DWSIM_Flowsheet.dwxmz`.
4. Press `F5` to run the flowsheet manually.

### 3. Training the Models (Optional)
The pre-trained models are included in `data/models/`. To execute the training pipeline from scratch:
1. Open a terminal in the root directory.
2. Train the Linear Regression Baseline:
   ```bash
   python Code/train_lr.py
   ```
3. Train the XGBoost Tuner:
   ```bash
   python Code/train_xgb.py
   ```
4. Train the PyTorch Physics-Informed Neural Network:
   ```bash
   python Code/train_ann.py
   ```

### 4. Explainable AI and Physics Verification
To verify the physical consistency of the models, execute the explanation script which generates SHAP feature importance charts and Parametric Sensitivity trend lines:
```bash
python Code/explain.py
```
The output plots will be saved to `data/plots/`.

### 5. Running the Interactive Streamlit Web UI
To interact with the Surrogate Model in real-time using UI sliders, run the Streamlit application:
```bash
streamlit run app.py
```
This will automatically launch a local web server in the browser.

## Assumptions
- Feed flow rate is assumed constant at 100 mol/s. 
- The distillation column operates with a total condenser and partial reboiler.
- The thermodynamic property package is Peng-Robinson (PR).
- Data samples that failed to converge in DWSIM or violated the Law of Conservation of Mass were removed during preprocessing to maintain dataset integrity.
