import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- Page Configuration ---
st.set_page_config(page_title="DWSIM Surrogate Model", page_icon="⚗️", layout="wide")

# --- Load Models & Scalers ---
@st.cache_resource
def load_assets():
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "models")
    xgb_model = joblib.load(os.path.join(models_dir, "xgb_model.pkl"))
    scaler_X = joblib.load(os.path.join(models_dir, "scaler_X.pkl"))
    scaler_y = joblib.load(os.path.join(models_dir, "scaler_y.pkl"))
    return xgb_model, scaler_X, scaler_y

try:
    model, scaler_X, scaler_y = load_assets()
except Exception as e:
    st.error(f"Error loading models. Have you trained them yet? \n\n Details: {e}")
    st.stop()

# --- Main UI ---
# Premium Custom CSS
st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Subtle Gradient Background */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        color: #e2e8f0;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: rgba(15, 23, 42, 0.7);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Headings */
    h1, h2, h3 {
        background: -webkit-linear-gradient(45deg, #60a5fa, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700 !important;
    }
    
    /* Metric Cards Glassmorphism */
    div[data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease, background 0.3s ease;
    }
    
    div[data-testid="metric-container"]:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.3);
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(96, 165, 250, 0.5);
    }
    
    div[data-testid="metric-container"] label {
        color: #94a3b8 !important;
        font-weight: 600;
        font-size: 1rem;
    }
    
    div[data-testid="metric-container"] div {
        color: #f8fafc !important;
        font-weight: 700;
    }
    
    /* Sliders styling */
    .stSlider > div > div > div {
        background-color: #3b82f6 !important;
    }
    
    /* Dividers */
    hr {
        border-color: rgba(255, 255, 255, 0.1) !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("⚗️ Binary Distillation Surrogate Model")
st.markdown("""
<div style="background: rgba(255, 255, 255, 0.05); padding: 15px; border-radius: 10px; border-left: 4px solid #60a5fa; margin-bottom: 20px;">
    Welcome to the interactive <b>Benzene-Toluene Distillation</b> surrogate model. <br>
    This dashboard uses an <b>XGBoost Regressor (99.6% Accuracy)</b> trained on 5,000 rigorous DWSIM simulations to instantly predict product purities and energy requirements.
</div>
""", unsafe_allow_html=True)

st.divider()

# --- Sidebar Inputs ---
st.sidebar.header("⚙️ Column Operating Conditions")
st.sidebar.markdown("Adjust the sliders below to see real-time predictions.")

# Default values based on the training data ranges
feed_temp = st.sidebar.slider("Feed Temperature (K)", min_value=290.0, max_value=350.0, value=300.0, step=1.0)
feed_press = st.sidebar.slider("Feed Pressure (Pa)", min_value=100000.0, max_value=150000.0, value=101325.0, step=100.0)
benzene_frac = st.sidebar.slider("Benzene Feed Fraction (z)", min_value=0.2, max_value=0.8, value=0.5, step=0.01)
stages = st.sidebar.slider("Number of Stages", min_value=10, max_value=30, value=20, step=1)
feed_stage = st.sidebar.slider("Feed Stage Location", min_value=5, max_value=15, value=10, step=1)
reflux_ratio = st.sidebar.slider("Reflux Ratio", min_value=1.0, max_value=5.0, value=2.5, step=0.1)
bottoms_rate = st.sidebar.slider("Bottoms Withdrawal Rate (mol/s)", min_value=40.0, max_value=60.0, value=50.0, step=0.5)

# --- Prediction Logic ---
# Pack inputs into a dataframe that matches training feature names
input_data = pd.DataFrame({
    "Feed_Temp_K": [feed_temp],
    "Feed_Press_Pa": [feed_press],
    "Benzene_Frac": [benzene_frac],
    "Stages": [stages],
    "Feed_Stage": [feed_stage],
    "Reflux_Ratio": [reflux_ratio],
    "Bottoms_Rate": [bottoms_rate]
})

# Scale inputs
scaled_input = scaler_X.transform(input_data)

# Predict
scaled_prediction = model.predict(scaled_input)

# Unscale predictions
final_prediction = scaler_y.inverse_transform(scaled_prediction)[0]

xD = final_prediction[0]
xB = final_prediction[1]
Qc = final_prediction[2]
Qr = final_prediction[3]

# --- Display Results ---
st.subheader("📊 Instant Predictions")

col1, col2 = st.columns(2)

with col1:
    st.info("### 💧 Product Purities")
    st.metric("Distillate Benzene Purity ($x_D$)", f"{xD:.4f}")
    st.metric("Bottoms Benzene Purity ($x_B$)", f"{xB:.4f}")

with col2:
    st.success("### ⚡ Energy Duties")
    st.metric("Condenser Duty ($Q_C$)", f"{Qc:.2f} kW")
    st.metric("Reboiler Duty ($Q_R$)", f"{Qr:.2f} kW")

st.divider()

st.markdown("""
<div style="text-align: center; color: gray;">
    <small>Built for FOSSEE Autumn 2026 Screening Task</small>
</div>
""", unsafe_allow_html=True)
