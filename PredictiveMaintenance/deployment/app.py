
import streamlit as st
import pandas as pd
import joblib
import os
from huggingface_hub import hf_hub_download

#  SET UP THE PAGE 
st.set_page_config(page_title="Engine Health Predictor", layout="centered")

# Hugging Face model details
repo_id = "SantoshS23/PredMaintModel"
filename = "adaboost_predictive_maintenance_model.joblib"

# Get Hugging Face token from environment variables
hf_token = os.getenv("HF_TOKEN")

@st.cache_resource
def load_model(repo_id, filename, token):
    """Caches the model loading process."""
    if not token:
        st.error("HF_TOKEN environment variable not set. Cannot load model.")
        st.info("Please set the HF_TOKEN environment variable with your Hugging Face access token.")
        st.stop()
    try:
        model_path = hf_hub_download(repo_id=repo_id, filename=filename, token=token)
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model from Hugging Face Hub: {e}")
        st.info("Please ensure you have set the HF_TOKEN environment variable with a valid token.")
        st.stop()

model = load_model(repo_id, filename, hf_token)

if model:
    st.success("Model loaded successfully from Hugging Face Hub!")
else:
    st.error("Model could not be loaded. Please check your HF_TOKEN and repo details.")
    st.stop()


# 3. APP HEADER
st.title("🛡️ Engine Predictive Maintenance")
st.markdown("""
Enter the current sensor telemetry below to predict the likelihood of engine failure.
This tool utilizes an optimized machine learning model to detect failure signatures.
""")

st.divider()

# 4. INPUT SECTION
st.subheader("📡 Real-Time Sensor Telemetry")

col1, col2 = st.columns(2)

with col1:
    rpm = st.number_input("Engine RPM", min_value=0.0, max_value=3000.0, value=750.0, step=10.0)
    lub_oil_press = st.number_input("Lub Oil Pressure (bar)", min_value=0.0, max_value=15.0, value=3.5, step=0.1)
    fuel_press = st.number_input("Fuel Pressure (bar)", min_value=0.0, max_value=25.0, value=6.0, step=0.1)

with col2:
    coolant_press = st.number_input("Coolant Pressure (bar)", min_value=0.0, max_value=15.0, value=2.5, step=0.1)
    lub_oil_temp = st.number_input("Lub Oil Temp (°C)", min_value=0.0, max_value=150.0, value=78.0, step=0.5)
    coolant_temp = st.number_input("Coolant Temp (°C)", min_value=0.0, max_value=200.0, value=82.0, step=0.5)

st.divider()

# 5. PREDICTION LOGIC
# Ensure the feature order matches exactly what the model was trained on
features = np.array([[rpm, lub_oil_press, fuel_press, coolant_press, lub_oil_temp, coolant_temp]])

if st.button("Analyze Engine Health"):
    prediction = model.predict(features)
    probability = model.predict_proba(features)[0][1] # Probability of failure (class 1)

    st.subheader("🔍 Analysis Result")

    if prediction[0] == 0:
        st.success(f"**NORMAL OPERATION**: The engine is healthy. (Confidence: {1-probability:.2%})")
        st.balloons()
    else:
        st.error(f"**FAULT DETECTED**: Maintenance required. High risk of failure! (Confidence: {probability:.2%})")

    # Additional Context for the User
    st.info(f"""
    **Expert Insight**:
    - Predicted Probabilty of Failure: **{probability:.4f}**
    - Alert Threshold: **0.50**
    """)

# 6. FOOTER/INFO
with st.expander("About this model"):
    st.write("""
    This model was trained using a gradient-boosted decision tree algorithm (XGBoost)
    optimized for engine sensor telemetry. It identifies non-linear correlations
    between pressure, temperature, and speed to predict potential mechanical failures
    before they occur.
    """)

