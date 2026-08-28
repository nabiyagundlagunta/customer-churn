```python
import streamlit as st
import pandas as pd
import joblib
import os

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# --------------------------------------------------
# LOAD ML MODEL
# --------------------------------------------------

MODEL_PATH = "churn_model.pkl"

if not os.path.exists(MODEL_PATH):
    st.error("❌ ML model file not found!")
    st.write(f"Expected model file: `{MODEL_PATH}`")
    st.stop()

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    st.error("❌ Unable to load ML model.")
    st.exception(e)
    st.stop()

# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("📊 Customer Churn Prediction")
st.write(
    "Enter the customer's details below to predict whether "
    "the customer is likely to churn."
)

st.divider()

# --------------------------------------------------
# CUSTOMER DETAILS
# --------------------------------------------------

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    senior_citizen = st.selectbox(
        "Senior Citizen",
        [0, 1]
    )

    partner = st.selectbox(
        "Partner",
        ["Yes", "No"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["Yes", "No"]
    )

    tenure = st.number_input(
        "Tenure (months)",
        min_value=0,
        max_value=100,
        value=12
    )

    phone_service = st.selectbox(
        "Phone Service",
        ["Yes", "No"]
    )

with col2:
    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )

    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=70.0
    )

    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=1000.0
    )

st.divider()

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if st.button("🔮 Predict Churn", use_container_width=True):

    # Create input dataframe
    input_data = pd.DataFrame({
        "gender": [gender],
        "SeniorCitizen": [senior_citizen],
        "Partner": [partner],
        "Dependents": [dependents],
        "tenure": [tenure],
        "PhoneService": [phone_service],
        "InternetService": [internet_service],
        "Contract": [contract],
        "PaperlessBilling": [paperless_billing],
        "PaymentMethod": [payment_method],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges]
    })

    try:
        prediction = model.predict(input_data)

        if prediction[0] == 1 or str(prediction[0]).lower() in ["yes", "churn", "true"]:
            st.error("⚠️ Customer is likely to CHURN.")
        else:
            st.success("✅ Customer is likely to STAY.")

        # Probability, if the model supports it
        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(input_data)

            churn_probability = probability[0][1] * 100

            st.metric(
                "Churn Probability",
                f"{churn_probability:.2f}%"
            )

    except Exception as e:
        st.error("❌ Prediction failed.")
        st.exception(e)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Customer Churn Prediction | Machine Learning Project"
)
```
