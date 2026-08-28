import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import io
from datetime import datetime

# PDF
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
    .main {
        padding-top: 1rem;
    }

    .title {
        font-size: 42px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #666;
        margin-bottom: 30px;
    }

    .result-box {
        padding: 25px;
        border-radius: 15px;
        margin-top: 20px;
        text-align: center;
        border: 1px solid #ddd;
    }

    .churn {
        background-color: #ffe5e5;
    }

    .safe {
        background-color: #e5ffe9;
    }

    .metric-box {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #ddd;
        text-align: center;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="title">📊 Customer Churn Prediction System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Predict whether a customer is likely to leave the company</div>',
    unsafe_allow_html=True
)

# ============================================================
# MODEL LOADING
# ============================================================

MODEL_PATHS = [
    "churn_model.pkl",
    "model.pkl",
    "customer_churn_model.pkl",
    "best_model.pkl"
]

SCALER_PATHS = [
    "scaler.pkl",
    "standard_scaler.pkl"
]

model = None
scaler = None


def load_first_existing(paths):
    for path in paths:
        if os.path.exists(path):
            try:
                return joblib.load(path), path
            except Exception:
                pass
    return None, None


model, model_path = load_first_existing(MODEL_PATHS)
scaler, scaler_path = load_first_existing(SCALER_PATHS)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚙️ Navigation")

page = st.sidebar.radio(
    "Choose a section:",
    [
        "🏠 Home",
        "🔮 Single Prediction",
        "📂 Batch Prediction",
        "📈 Data Analysis",
        "ℹ️ About Model"
    ]
)

st.sidebar.markdown("---")

if model_path:
    st.sidebar.success(f"Model loaded: {model_path}")
else:
    st.sidebar.warning(
        "Model file not found. Please upload a trained model."
    )


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def convert_yes_no(value):
    """Convert common Yes/No values to 1/0."""
    if isinstance(value, str):
        value = value.strip().lower()

        if value in ["yes", "y", "true", "1"]:
            return 1

        if value in ["no", "n", "false", "0"]:
            return 0

    return value


def prepare_features(df):
    """
    Prepare uploaded/customer data for model prediction.

    Handles common Telco Customer Churn dataset columns.
    """

    data = df.copy()

    # Remove customer ID
    id_columns = [
        "customerID",
        "customer_id",
        "CustomerID",
        "Customer Id"
    ]

    for col in id_columns:
        if col in data.columns:
            data = data.drop(columns=[col])

    # Target column should not be passed to model
    target_columns = [
        "Churn",
        "churn",
        "target",
        "Target"
    ]

    for col in target_columns:
        if col in data.columns:
            data = data.drop(columns=[col])

    # Convert TotalCharges to numeric
    if "TotalCharges" in data.columns:
        data["TotalCharges"] = pd.to_numeric(
            data["TotalCharges"],
            errors="coerce"
        )

        data["TotalCharges"] = data["TotalCharges"].fillna(
            data["TotalCharges"].median()
        )

    # Convert common Yes/No columns
    for col in data.columns:
        if data[col].dtype == "object":

            unique_values = set(
                str(x).strip().lower()
                for x in data[col].dropna().unique()
            )

            if unique_values.issubset(
                {"yes", "no", "y", "n", "true", "false", "0", "1"}
            ):
                data[col] = data[col].apply(convert_yes_no)

    # One-hot encode remaining categorical columns
    categorical_columns = data.select_dtypes(
        include=["object", "category"]
    ).columns

    if len(categorical_columns) > 0:
        data = pd.get_dummies(
            data,
            columns=categorical_columns,
            drop_first=True
        )

    # Replace missing values
    data = data.replace([np.inf, -np.inf], np.nan)

    data = data.fillna(0)

    return data


def align_features(data):
    """
    Align input columns with model expected features
    when feature names are available.
    """

    if model is None:
        return data

    if hasattr(model, "feature_names_in_"):

        expected = list(model.feature_names_in_)

        # Add missing columns
        for column in expected:
            if column not in data.columns:
                data[column] = 0

        # Remove extra columns
        data = data[expected]

    return data


def make_prediction(input_df):
    """Run model prediction."""

    if model is None:
        return None, None

    prepared = prepare_features(input_df)

    prepared = align_features(prepared)

    # Apply scaler if available
    if scaler is not None:

        try:

            numeric_columns = prepared.select_dtypes(
                include=[np.number]
            ).columns

            if len(numeric_columns) > 0:

                prepared[numeric_columns] = scaler.transform(
                    prepared[numeric_columns]
                )

        except Exception:
            # If scaler doesn't match, continue without it
            pass

    prediction = model.predict(prepared)

    probability = None

    if hasattr(model, "predict_proba"):

        try:
            probability = model.predict_proba(prepared)[:, 1]
        except Exception:
            probability = None

    return prediction, probability


def get_risk_level(probability):
    """Return risk category."""

    if probability is None:
        return "Unknown"

    if probability >= 0.70:
        return "High Risk"

    if probability >= 0.40:
        return "Medium Risk"

    return "Low Risk"


def get_recommendations(probability):
    """Generate simple customer retention recommendations."""

    if probability is None:
        return [
            "Review the customer profile manually.",
            "Monitor customer activity regularly."
        ]

    if probability >= 0.70:
        return [
            "Contact the customer proactively.",
            "Offer a suitable retention plan.",
            "Provide personalized support.",
            "Check whether pricing or service issues exist."
        ]

    if probability >= 0.40:
        return [
            "Monitor customer engagement.",
            "Provide personalized offers.",
            "Check satisfaction levels.",
            "Encourage long-term subscription."
        ]

    return [
        "Continue normal customer engagement.",
        "Maintain good service quality.",
        "Offer loyalty benefits when appropriate."
    ]


def create_pdf(result_data):
    """Create a PDF report."""

    buffer = io.BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4
    )

    styles = getSampleStyleSheet()

    story = []

    story.append(
        Paragraph(
            "Customer Churn Prediction Report",
            styles["Title"]
        )
    )

    story.append(Spacer(1, 15))

    rows = [
        ["Field", "Value"]
    ]

    for key, value in result_data.items():
        rows.append(
            [
                str(key),
                str(value)
            ]
        )

    table = Table(rows, colWidths=[180, 300])

    table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.lightgrey
                ),
                (
                    "TEXTCOLOR",
                    (0, 0),
                    (-1, 0),
                    colors.black
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    1,
                    colors.grey
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold"
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "TOP"
                )
            ]
        )
    )

    story.append(table)

    story.append(Spacer(1, 20))

    story.append(
        Paragraph(
            "Generated by Customer Churn Prediction System",
            styles["Normal"]
        )
    )

    story.append(
        Paragraph(
            datetime.now().strftime(
                "Generated on: %d-%m-%Y %H:%M"
            ),
            styles["Normal"]
        )
    )

    document.build(story)

    buffer.seek(0)

    return buffer


# ============================================================
# HOME
# ============================================================

if page == "🏠 Home":

    st.header("Welcome 👋")

    st.write(
        """
        This application uses machine learning to predict
        whether a customer is likely to churn.
        """
    )

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Prediction",
            "AI Based"
        )

    with col2:
        st.metric(
            "Application",
            "Customer Churn"
        )

    with col3:
        st.metric(
            "Interface",
            "Streamlit"
        )

    st.markdown("---")

    st.subheader("How it works")

    st.write(
        """
        1. Enter customer information.
        2. The application prepares the input data.
        3. The trained machine learning model analyzes the customer.
        4. The application predicts churn probability.
        5. A risk level and retention recommendations are displayed.
        """
    )

    st.info(
        "Go to 'Single Prediction' from the sidebar to test a customer."
    )


# ============================================================
# SINGLE PREDICTION
# ============================================================

elif page == "🔮 Single Prediction":

    st.header("🔮 Single Customer Prediction")

    if model is None:

        st.error(
            """
            Model file was not found.

            Upload one of these files to your GitHub/Streamlit project:

            - churn_model.pkl
            - model.pkl
            - customer_churn_model.pkl
            - best_model.pkl
            """
        )

    else:

        st.write(
            "Enter the customer details below."
        )

        with st.form("customer_form"):

            col1, col2 = st.columns(2)

            with col1:

                gender = st.selectbox(
                    "Gender",
                    ["Male", "Female"]
                )

                senior_citizen = st.selectbox(
                    "Senior Citizen",
                    ["No", "Yes"]
                )

                partner = st.selectbox(
                    "Partner",
                    ["No", "Yes"]
                )

                dependents = st.selectbox(
                    "Dependents",
                    ["No", "Yes"]
                )

                tenure = st.number_input(
                    "Tenure (months)",
                    min_value=0,
                    max_value=100,
                    value=12
                )

                phone_service = st.selectbox(
                    "Phone Service",
                    ["No", "Yes"]
                )

                multiple_lines = st.selectbox(
                    "Multiple Lines",
                    [
                        "No phone service",
                        "No",
                        "Yes"
                    ]
                )

            with col2:

                internet_service = st.selectbox(
                    "Internet Service",
                    [
                        "DSL",
                        "Fiber optic",
                        "No"
                    ]
                )

                online_security = st.selectbox(
                    "Online Security",
                    [
                        "No",
                        "Yes",
                        "No internet service"
                    ]
                )

                online_backup = st.selectbox(
                    "Online Backup",
                    [
                        "No",
                        "Yes",
                        "No internet service"
                    ]
                )

                device_protection = st.selectbox(
                    "Device Protection",
                    [
                        "No",
                        "Yes",
                        "No internet service"
                    ]
                )

                tech_support = st.selectbox(
                    "Tech Support",
                    [
                        "No",
                        "Yes",
                        "No internet service"
                    ]
                )

                streaming_tv = st.selectbox(
                    "Streaming TV",
                    [
                        "No",
                        "Yes",
                        "No internet service"
                    ]
                )

                streaming_movies = st.selectbox(
                    "Streaming Movies",
                    [
                        "No",
                        "Yes",
                        "No internet service"
                    ]
                )

            col3, col4 = st.columns(2)

            with col3:

                contract = st.selectbox(
                    "Contract",
                    [
                        "Month-to-month",
                        "One year",
                        "Two year"
                    ]
                )

                paperless_billing = st.selectbox(
                    "Paperless Billing",
                    [
                        "No",
                        "Yes"
                    ]
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

            with col4:

                monthly_charges = st.number_input(
                    "Monthly Charges",
                    min_value=0.0,
                    value=70.0
                )

                total_charges = st.number_input(
                    "Total Charges",
                    min_value=0.0,
                    value=800.0
                )

            submitted = st.form_submit_button(
                "🚀 Predict Churn"
            )

        if submitted:

            customer = pd.DataFrame(
                {
                    "gender": [gender],
                    "SeniorCitizen": [
                        1 if senior_citizen == "Yes" else 0
                    ],
                    "Partner": [partner],
                    "Dependents": [dependents],
                    "tenure": [tenure],
                    "PhoneService": [phone_service],
                    "MultipleLines": [multiple_lines],
                    "InternetService": [internet_service],
                    "OnlineSecurity": [online_security],
                    "OnlineBackup": [online_backup],
                    "DeviceProtection": [device_protection],
                    "TechSupport": [tech_support],
                    "StreamingTV": [streaming_tv],
                    "StreamingMovies": [streaming_movies],
                    "Contract": [contract],
                    "PaperlessBilling": [paperless_billing],
                    "PaymentMethod": [payment_method],
                    "MonthlyCharges": [monthly_charges],
                    "TotalCharges": [total_charges]
                }
            )

            try:

                prediction, probability = make_prediction(
                    customer
                )

                if prediction is not None:

                    pred_value = int(prediction[0])

                    if probability is not None:
                        churn_probability = float(
                            probability[0]
                        )
                    else:
                        churn_probability = None

                    if pred_value == 1:

                        st.markdown(
                            """
                            <div class="result-box churn">
                            <h2>⚠️ Customer Likely to Churn</h2>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                        prediction_text = "Likely to Churn"

                    else:

                        st.markdown(
                            """
                            <div class="result-box safe">
                            <h2>✅ Customer Likely to Stay</h2>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                        prediction_text = "Likely to Stay"

                    st.markdown("---")

                    c1, c2, c3 = st.columns(3)

                    with c1:

                        st.metric(
                            "Prediction",
                            prediction_text
                        )

                    with c2:

                        if churn_probability is not None:

                            st.metric(
                                "Churn Probability",
                                f"{churn_probability * 100:.2f}%"
                            )

                    with c3:

                        risk = get_risk_level(
                            churn_probability
                        )

                        st.metric(
                            "Risk Level",
                            risk
                        )

                    # ------------------------------------------------
                    # Recommendations
                    # ------------------------------------------------

                    st.subheader(
                        "💡 Recommended Actions"
                    )

                    recommendations = get_recommendations(
                        churn_probability
                    )

                    for item in recommendations:
                        st.write(
                            f"• {item}"
                        )

                    # ------------------------------------------------
                    # Report
                    # ------------------------------------------------

                    report_data = {
                        "Prediction": prediction_text,
                        "Risk Level": risk,
                        "Churn Probability": (
                            f"{churn_probability * 100:.2f}%"
                            if churn_probability is not None
                            else "N/A"
                        ),
                        "Tenure": tenure,
                        "Monthly Charges": monthly_charges,
                        "Total Charges": total_charges,
                        "Contract": contract,
                        "Internet Service": internet_service
                    }

                    pdf_file = create_pdf(
                        report_data
                    )

                    st.download_button(
                        label="📄 Download PDF Report",
                        data=pdf_file,
                        file_name="customer_churn_report.pdf",
                        mime="application/pdf"
                    )

            except Exception as e:

                st.error(
                    "Prediction failed."
                )

                st.code(
                    str(e)
                )

                st.info(
                    """
                    This usually happens when the input columns do not
                    match the columns used while training the model.
                    """
                )


# ============================================================
# BATCH PREDICTION
# ============================================================

elif page == "📂 Batch Prediction":

    st.header("📂 Batch Customer Prediction")

    st.write(
        """
        Upload a CSV or Excel file containing multiple customers.
        """
    )

    uploaded_file = st.file_uploader(
        "Upload customer data",
        type=["csv", "xlsx"]
    )

    if uploaded_file is not None:

        try:

            if uploaded_file.name.endswith(".csv"):

                data = pd.read_csv(
                    uploaded_file
                )

            else:

                data = pd.read_excel(
                    uploaded_file
                )

            st.subheader("Uploaded Data")

            st.dataframe(
                data.head(20),
                use_container_width=True
            )

            if model is None:

                st.error(
                    "Please upload your trained model first."
                )

            else:

                if st.button(
                    "🚀 Predict All Customers"
                ):

                    try:

                        predictions, probabilities = make_prediction(
                            data
                        )

                        result = data.copy()

                        result["Prediction"] = np.where(
                            predictions == 1,
                            "Likely to Churn",
                            "Likely to Stay"
                        )

                        if probabilities is not None:

                            result["Churn Probability"] = (
                                probabilities * 100
                            ).round(2)

                            result["Risk Level"] = [
                                get_risk_level(x)
                                for x in probabilities
                            ]

                        st.success(
                            "Prediction completed successfully!"
                        )

                        st.dataframe(
                            result,
                            use_container_width=True
                        )

                        # CSV download

                        csv_data = result.to_csv(
                            index=False
                        ).encode("utf-8")

                        st.download_button(
                            "⬇️ Download CSV Results",
                            data=csv_data,
                            file_name="churn_predictions.csv",
                            mime="text/csv"
                        )

                        # Excel download

                        excel_buffer = io.BytesIO()

                        with pd.ExcelWriter(
                            excel_buffer,
                            engine="openpyxl"
                        ) as writer:

                            result.to_excel(
                                writer,
                                index=False,
                                sheet_name="Predictions"
                            )

                        excel_buffer.seek(0)

                        st.download_button(
                            "📊 Download Excel Results",
                            data=excel_buffer,
                            file_name="churn_predictions.xlsx",
                            mime=(
                                "application/vnd.openxmlformats-"
                                "officedocument.spreadsheetml.sheet"
                            )
                        )

                    except Exception as e:

                        st.error(
                            "Batch prediction failed."
                        )

                        st.code(
                            str(e)
                        )

        except Exception as e:

            st.error(
                "Could not read the uploaded file."
            )

            st.code(
                str(e)
            )


# ============================================================
# DATA ANALYSIS
# ============================================================

elif page == "📈 Data Analysis":

    st.header("📈 Customer Data Analysis")

    uploaded_file = st.file_uploader(
        "Upload CSV for analysis",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:

            df = pd.read_csv(
                uploaded_file
            )

            st.subheader("Dataset Preview")

            st.dataframe(
                df.head(20),
                use_container_width=True
            )

            st.markdown("---")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Rows",
                    df.shape[0]
                )

            with col2:
                st.metric(
                    "Columns",
                    df.shape[1]
                )

            with col3:
                st.metric(
                    "Missing Values",
                    int(df.isnull().sum().sum())
                )

            with col4:
                st.metric(
                    "Duplicate Rows",
                    int(df.duplicated().sum())
                )

            st.markdown("---")

            st.subheader("📊 Dataset Information")

            st.write(
                df.describe(include="all").T
            )

            # Churn distribution

            churn_column = None

            for col in [
                "Churn",
                "churn",
                "target",
                "Target"
            ]:

                if col in df.columns:

                    churn_column = col
                    break

            if churn_column:

                st.subheader(
                    "Customer Churn Distribution"
                )

                churn_counts = (
                    df[churn_column]
                    .value_counts()
                )

                st.bar_chart(
                    churn_counts
                )

        except Exception as e:

            st.error(
                "Could not analyze the dataset."
            )

            st.code(
                str(e)
            )

    else:

        st.info(
            "Upload a CSV file to start data analysis."
        )


# ============================================================
# ABOUT MODEL
# ============================================================

elif page == "ℹ️ About Model":

    st.header("ℹ️ About Customer Churn Prediction")

    st.write(
        """
        ### What is Customer Churn?

        Customer churn occurs when a customer stops using a company's
        products or services.

        ### Objective

        The objective of this project is to use machine learning to
        identify customers who have a higher probability of leaving.

        ### Common Features

        The model may use information such as:

        - Customer tenure
        - Monthly charges
        - Total charges
        - Contract type
        - Internet service
        - Payment method
        - Technical support
        - Online security
        - Streaming services
        - Customer demographics

        ### Benefits

        Early churn prediction can help businesses:

        - Identify high-risk customers
        - Improve customer retention
        - Provide personalized offers
        - Reduce customer acquisition costs
        - Improve customer satisfaction
        """
    )

    st.markdown("---")

    st.subheader("Model Status")

    if model is not None:

        st.success(
            f"Model successfully loaded: {model_path}"
        )

        if hasattr(model, "feature_names_in_"):

            st.write(
                "Number of expected features:",
                len(model.feature_names_in_)
            )

    else:

        st.warning(
            "No model file detected."
        )

    if scaler is not None:

        st.success(
            f"Scaler loaded: {scaler_path}"
        )

    else:

        st.info(
            "No separate scaler found."
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Customer Churn Prediction System | Built with Python, "
    "Pandas, Scikit-learn and Streamlit"
)
