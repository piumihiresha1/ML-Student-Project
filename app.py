import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ******************************************
# 1. Load the Saved Model Assets (The Logistic Regression Model, Standard Scaler, and Label Encoder
# ******************************************

@st.cache_resource
def load_assets():
    with open('best_model.pkl', 'rb') as file:
        assets = pickle.load(file)
    return assets['model'], assets['scaler'], assets['encoder']

model, scaler, encoder = load_assets()

# ******************************************
# 2. Build the Streamlit User Interface
# ******************************************
st.set_page_config(page_title="Churn Predictor", page_icon="📉", layout="centered")

st.title("📉 Customer Churn Prediction App")
st.write("Enter the customer's billing and account details below to predict if they are at risk of churning.")
st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Numeric Features")
    tenure = st.number_input("Tenure (Months)", min_value=0.0, value=12.0, step=1.0)
    monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=1500.0)
    total_charges = st.number_input("Total Charges", min_value=0.0, value=18000.0)
    support_calls = st.number_input("Support Calls Made", min_value=0.0, value=1.0, step=1.0)

with col2:
    st.subheader("Categorical Features")
    # Using the exact categories found in your dataset
    contract_type = st.selectbox("Contract Type", ["Prepaid", "Postpaid"])
    payment_method = st.selectbox("Payment Method", ["Credit Card", "Cash", "Bank Transfer", "Cheque"])
    dtv_sachet_waiver_active = st.selectbox("DTV Sachet Waiver Active", ["Yes", "No"])

st.divider()

# ******************************************
# 3. Prediction Logic
# ******************************************
if st.button("🔮 Predict Churn Risk", use_container_width=True):
    
    # a. Gather all inputs into a single-row DataFrame
    input_data = pd.DataFrame({
        'TENURE': [tenure],
        'MONTHLY_CHARGES': [monthly_charges],
        'TOTAL_CHARGES': [total_charges],
        'CONTRACT_TYPE': [contract_type],
        'PAYMENT_METHOD': [payment_method],
        'DTV_SACHET_WAIVER_ACTIVE': [dtv_sachet_waiver_active],
        'SUPPORT_CALLS': [support_calls]
    })
    
    # b. Encode the categorical text into numbers using your saved encoder
    contract_mapping = {"Postpaid": 0, "Prepaid": 1}
    payment_mapping = {"Bank Transfer": 0, "Cash": 1, "Cheque": 2, "Credit Card": 3}
    waiver_mapping = {"No": 0, "Yes": 1}

    input_data['CONTRACT_TYPE'] = input_data['CONTRACT_TYPE'].map(contract_mapping)
    input_data['PAYMENT_METHOD'] = input_data['PAYMENT_METHOD'].map(payment_mapping)
    input_data['DTV_SACHET_WAIVER_ACTIVE'] = input_data['DTV_SACHET_WAIVER_ACTIVE'].map(waiver_mapping)

    # c. Scale the data using the saved StandardScaler
    input_scaled = scaler.transform(input_data)
    
    # d. Make the prediction
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1] 
    
    # e. Display Results
    st.subheader("Prediction Results:")
    if prediction == 1:
        st.error(f"⚠️ **HIGH RISK:** This customer is likely to CHURN.")
        st.write(f"**Churn Probability:** {probability:.2%}")
        st.info("💡 **Actionable Insight:** Consider offering a retention discount or reaching out to address their support tickets.")
    else:
        st.success(f"✅ **SAFE:** This customer is likely to be RETAINED.")
        st.write(f"**Retention Probability:** {(1 - probability):.2%}")
