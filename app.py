import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ==========================================
# 1. Load the Saved Model Assets
# ==========================================
# @st.cache_resource ensures the model only loads once and stays in memory
@st.cache_resource
def load_assets():
    with open('best_model.pkl', 'rb') as file:
        assets = pickle.load(file)
    return assets['model'], assets['scaler'], assets['encoder']

# Load the logistic regression model, standard scaler, and label encoder
model, scaler, encoder = load_assets()

# ==========================================
# 2. Build the Streamlit User Interface
# ==========================================
st.set_page_config(page_title="Churn Predictor", page_icon="📉", layout="centered")

st.title("📉 Customer Churn Prediction App")
st.write("Enter the customer's billing and account details below to predict if they are at risk of churning.")

st.divider()

# Create two columns for a cleaner layout
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

# ==========================================
# 3. Prediction Logic
# ==========================================
if st.button("🔮 Predict Churn Risk", use_container_width=True):
    
    # 1. Gather all inputs into a single-row DataFrame (must match the exact column names/order used in training)
    input_data = pd.DataFrame({
        'TENURE': [tenure],
        'MONTHLY_CHARGES': [monthly_charges],
        'TOTAL_CHARGES': [total_charges],
        'CONTRACT_TYPE': [contract_type],
        'PAYMENT_METHOD': [payment_method],
        'DTV_SACHET_WAIVER_ACTIVE': [dtv_sachet_waiver_active],
        'SUPPORT_CALLS': [support_calls]
    })
    
    # 2. Encode the categorical text into numbers using your saved encoder
    # Note: If you used a single LabelEncoder in a loop in your notebook, 
    # it might only remember the last column. If this throws an error, you can use manual mapping.
    categorical_cols = ['CONTRACT_TYPE', 'PAYMENT_METHOD', 'DTV_SACHET_WAIVER_ACTIVE']
    try:
        for col in categorical_cols:
            # We fit_transform here as a fallback in case the encoder wasn't saved as a dictionary
            input_data[col] = encoder.fit_transform(input_data[col]) 
    except Exception as e:
        st.error(f"Encoding error: {e}")

    # 3. Scale the data using the saved StandardScaler
    # Ensure column order matches the X_train dataset exactly
    input_scaled = scaler.transform(input_data)
    
    # 4. Make the prediction
    prediction = model.predict(input_scaled)[0]
    
    # Use predict_proba to get the confidence percentage (useful for business context)
    probability = model.predict_proba(input_scaled)[0][1] 
    
    # 5. Display beautiful results
    st.subheader("Prediction Results:")
    if prediction == 1:
        st.error(f"⚠️ **HIGH RISK:** This customer is likely to CHURN.")
        st.write(f"**Churn Probability:** {probability:.2%}")
        st.info("💡 **Actionable Insight:** Consider offering a retention discount or reaching out to address their support tickets.")
    else:
        st.success(f"✅ **SAFE:** This customer is likely to be RETAINED.")
        st.write(f"**Retention Probability:** {(1 - probability):.2%}")
