import streamlit as st
import numpy as np
import pickle

st.title("Telecom Churn Prediction Dashboard")
st.write("Analyze subscriber metadata to predict the likelihood of service cancellation.")

# Load Model
try:
    with open('best_model.pkl', 'rb') as file:
        data = pickle.load(file)
        model = data['model']
        scaler = data['scaler']
except FileNotFoundError:
    st.error("Model file not found. Ensure best_model.pkl is in the repository.")
    st.stop()

# UI Inputs
col1, col2 = st.columns(2)
with col1:
    tenure = st.number_input("Tenure (Months)", 0, 120, 12)
    monthly_charges = st.number_input("Monthly Charges (LKR)", 0.0, 10000.0, 1500.0)
    contract_type = st.selectbox("Contract Type", [0, 1, 2], format_func=lambda x: ['Prepaid', 'Postpaid 1YR', 'Postpaid 2YR'][x])
with col2:
    sachet_waiver = st.selectbox("DTV Sachet Waiver Active?", [0, 1], format_func=lambda x: ['No', 'Yes'][x])
    total_charges = st.number_input("Total Charges (LKR)", 0.0, value=18000.0)
    support_calls = st.number_input("Support Calls (Last 30 Days)", 0, 20, 1)

# Prediction
if st.button("Predict Churn Risk"):
    input_data = np.array([[tenure, monthly_charges, contract_type, sachet_waiver, total_charges, support_calls]])
    input_scaled = scaler.transform(input_data)
    
    prediction = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]
    
    if prediction == 1:
        st.error(f"⚠️ High Risk of Churn! (Probability: {prob:.1%})")
    else:
        st.success(f"✅ Low Risk. (Probability: {prob:.1%})")