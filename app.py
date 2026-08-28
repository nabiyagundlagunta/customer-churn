import streamlit as st
import streamlit.components.v1 as components
import joblib
import pandas as pd
import base64
import json
import zlib
import os
import uuid
from openai import OpenAI
from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate,
    Table,
    TableStyle,
    Paragraph,
    Spacer
)


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Customer Retention Intelligence",
    page_icon="📊",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

/* =========================================================
   GENERAL FONT SIZES
   ========================================================= */

html, body, [class*="css"] {
    font-size: 18px;
}


/* =========================================================
   MAIN TITLE
   ========================================================= */

.main-title {
    font-size: 42px;
    font-weight: 700;
    line-height: 1.2;
    margin-bottom: 8px;
}


/* =========================================================
   SUBTITLE
   ========================================================= */

.subtitle {
    font-size: 20px;
    line-height: 1.5;
    color: #9ca3af;
    margin-bottom: 32px;
}


/* =========================================================
   SECTION TITLE
   ========================================================= */

.section-title {
    font-size: 28px;
    font-weight: 650;
    line-height: 1.3;
    margin-top: 28px;
    margin-bottom: 20px;
}


/* =========================================================
   NORMAL TEXT
   ========================================================= */

.stMarkdown p {
    font-size: 18px;
    line-height: 1.6;
}


/* =========================================================
   FIELD LABELS
   ========================================================= */

label {
    font-size: 18px !important;
    font-weight: 600 !important;
}


/* =========================================================
   TEXT INPUTS
   ========================================================= */

.stTextInput > div > div > input {
    border-radius: 10px;
    padding: 11px 14px;
    font-size: 18px;
    min-height: 48px;
}


/* =========================================================
   SELECT BOX
   ========================================================= */

.stSelectbox > div > div {
    border-radius: 10px;
    font-size: 17px;
}

[data-baseweb="select"] {
    font-size: 18px;
}


/* =========================================================
   NORMAL BUTTONS
   ========================================================= */

.stButton > button {
    border-radius: 12px;
    min-height: 52px;
    font-size: 18px;
    font-weight: 600;
    width: 100%;
}


/* =========================================================
   PREDICTION TYPE BUTTONS
   ========================================================= */

.prediction-option button {
    min-height: 165px !important;
    height: 165px !important;

    border-radius: 18px !important;

    font-size: 22px !important;
    font-weight: 650 !important;

    white-space: pre-wrap !important;

    line-height: 1.6 !important;

    padding: 25px !important;
}


/* =========================================================
   MODEL ONLINE BADGE
   ========================================================= */
.model-online-badge {
    position: fixed !important;
    top: 20px !important;
    right: 24px !important;
    z-index: 1000002 !important;
    display: inline-flex !important;
    align-items: center;
    gap: 9px;
    margin: 0 !important;
    padding: 8px 15px;
    border: 1px solid rgba(34, 197, 94, 0.35);
    border-radius: 999px;
    background: rgba(17, 24, 39, 0.96);
    color: #bbf7d0;
    font-size: 15px;
    font-weight: 700;
    letter-spacing: .2px;
    white-space: nowrap;
    box-shadow: 0 4px 16px rgba(0,0,0,.18);
}
.model-online-dot {
    width: 9px;
    height: 9px;
    border-radius: 50%;
    background: #22c55e;
    box-shadow: 0 0 0 4px rgba(34,197,94,.12);
    flex-shrink: 0;
}

/* =========================================================
   PAGE SCROLL FIX
   ========================================================= */
html, body {
    min-height: 100% !important;
    height: auto !important;
    overflow-x: hidden !important;
    overflow-y: auto !important;
}

[data-testid="stAppViewContainer"] {
    min-height: 100vh !important;
    height: auto !important;
    overflow-x: hidden !important;
    overflow-y: auto !important;
}

[data-testid="stMain"] {
    min-height: 100vh !important;
    height: auto !important;
    overflow: visible !important;
}

[data-testid="stMainBlockContainer"] {
    min-height: 100vh !important;
    height: auto !important;
    overflow: visible !important;
    padding-bottom: 80px !important;
}

[data-testid="stMainBlockContainer"] > div {
    height: auto !important;
    min-height: 0 !important;
    overflow: visible !important;
}

/* =========================================================
   SHARE RESULT CARD
   ========================================================= */
.share-card {
    margin-top: 24px;
    padding: 18px 20px;
    border: 1px solid #374151;
    border-radius: 16px;
    background: #111827;
}
.share-title {
    font-size: 19px;
    font-weight: 750;
    margin-bottom: 5px;
}
.share-subtitle {
    color: #9ca3af;
    font-size: 15px;
    line-height: 1.5;
}

/* =========================================================
   MOBILE BACK BUTTON
   ========================================================= */

/*
   Hidden on laptop/desktop.
   Shown only on smaller screens.
*/

.mobile-back-button {
    display: none;
}


/* =========================================================
   METRICS
   ========================================================= */

[data-testid="stMetric"] {
    background: #1f2937;
    padding: 20px;
    border-radius: 15px;
}

[data-testid="stMetricLabel"] {
    font-size: 18px !important;
}

[data-testid="stMetricValue"] {
    font-size: 29px !important;
}


/* =========================================================
   ALERTS
   ========================================================= */

.stAlert {
    border-radius: 12px;
    font-size: 18px;
}


/* =========================================================
   EXPANDER
   ========================================================= */

[data-testid="stExpander"] {
    font-size: 18px;
}


/* =========================================================
   DATAFRAME
   ========================================================= */

[data-testid="stDataFrame"] {
    font-size: 17px;
}


/* =========================================================
   REQUIRED FIELDS LIST
   ========================================================= */

.required-fields-list {
    margin: 10px 0 20px 0;
}

.required-fields-row {
    display: flex;
    gap: 14px;
    margin-bottom: 9px;
    flex-wrap: wrap;
}

.required-field {
    flex: 1 1 30%;
    min-width: 180px;
    padding: 9px 12px;
    border-radius: 8px;
    background: #1f2937;
    font-size: 16px;
    line-height: 1.4;
}


/* =========================================================
   REQUIRED FIELD HELP ICONS
   ========================================================= */

.required-field-content {
    display: inline-flex;
    align-items: center;
    gap: 7px;
}

.required-field-info {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 18px;
    height: 18px;
    border: 1px solid #9ca3af;
    border-radius: 50%;
    color: #d1d5db;
    font-size: 12px;
    font-weight: 700;
    line-height: 1;
    cursor: default;
    flex-shrink: 0;
}


/* =========================================================
   HAMBURGER NAVIGATION
   ========================================================= */

/*
   IMPORTANT:
   The custom navigation is rendered inside Streamlit's main content
   DOM. Streamlit can clip children at the main-content boundary.
   These rules explicitly allow the custom panel/button to extend
   into the blank top area of the app.
*/
[data-testid="stMain"],
[data-testid="stMainBlockContainer"],
[data-testid="stMainBlockContainer"] > div {
    overflow: visible !important;
}

/* Keep the actual Streamlit app container as the vertical scroll owner. */
[data-testid="stAppViewContainer"] {
    overflow-x: hidden !important;
    overflow-y: auto !important;
}

/* Closed menu: open button.
   IMPORTANT: st.markdown() and st.button() are separate Streamlit
   blocks, so the old .nav-hamburger-wrap could not position the
   actual button. Target the button's real keyed container instead. */
.st-key-hamburger_open {
    position: fixed !important;
    top: 36px !important;
    left: 18px !important;
    width: 52px !important;
    height: 52px !important;
    margin: 0 !important;
    padding: 0 !important;
    z-index: 1000000 !important;
    overflow: visible !important;
}

.st-key-hamburger_open > div,
.st-key-hamburger_open .stButton,
.st-key-hamburger_open .stButton > button {
    width: 52px !important;
    height: 52px !important;
    min-height: 52px !important;
}

.st-key-hamburger_open button,
.st-key-hamburger_close button {
    padding: 0 !important;
    margin: 0 !important;
    border-radius: 12px !important;
    font-size: 28px !important;
    line-height: 1 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    visibility: visible !important;
    opacity: 1 !important;
}

/* Open menu: full-height panel from the top of the app viewport. */
.st-key-nav_panel {
    position: fixed !important;
    top: 0 !important;
    left: 0 !important;
    bottom: 0 !important;
    width: 290px !important;
    height: 100vh !important;
    z-index: 999990 !important;

    margin: 0 !important;
    padding: 86px 18px 24px 18px !important;

    background: #111827 !important;
    border-right: 1px solid #374151 !important;
    box-shadow: 10px 0 30px rgba(0,0,0,.25) !important;

    /* Do not clip the fixed close button. */
    overflow: visible !important;
}

/*
   CLOSE BUTTON:
   It is independent of the panel's layout and sits above every
   Streamlit layer. This prevents the top portion from being hidden.
*/
.st-key-hamburger_close {
    position: fixed !important;
    top: 18px !important;
    left: 18px !important;
    width: 52px !important;
    height: 52px !important;
    margin: 0 !important;
    padding: 0 !important;
    z-index: 1000000 !important;
    overflow: visible !important;
}

.st-key-hamburger_close > div,
.st-key-hamburger_close .stButton {
    width: 52px !important;
    height: 52px !important;
    margin: 0 !important;
    padding: 0 !important;
}

.st-key-hamburger_close button {
    width: 52px !important;
    height: 52px !important;
    min-height: 52px !important;
    margin: 0 !important;
    padding: 0 !important;
    position: relative !important;
    z-index: 1000001 !important;
    visibility: visible !important;
    opacity: 1 !important;
}

/* Keep the navigation controls below the hamburger. */
.st-key-nav_prediction,
.st-key-nav_about {
    position: relative !important;
    z-index: 999995 !important;
}

.nav-panel-title { font-size: 22px; font-weight: 700; margin: 0 0 18px 8px; }
.nav-panel-note { color: #9ca3af; font-size: 14px; margin: 18px 8px 0 8px; line-height: 1.5; }
.gauge-row { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 18px; margin-top: 25px; }
.gauge-card { background: #1f2937; border: 1px solid #374151; border-radius: 18px; padding: 18px 12px 20px; text-align: center; }
.gauge-title { font-size: 17px; font-weight: 650; min-height: 45px; display: flex; align-items: center; justify-content: center; }
.gauge-svg { width: 100%; max-width: 240px; height: auto; display: block; margin: 4px auto -8px; }
.gauge-value { font-size: 30px; font-weight: 750; margin-top: -2px; }
.gauge-scale { color: #9ca3af; font-size: 13px; }
@media (max-width: 900px) { .gauge-row { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
@media (max-width: 520px) { .gauge-row { grid-template-columns: 1fr; } }

/* =========================================================
   MOBILE
   ========================================================= */

@media (max-width: 768px) {

    .required-field {
        min-width: 100%;
    }
}

/* =========================================================
   MOBILE RESPONSIVE DESIGN
   ========================================================= */

@media (max-width: 768px) {

    .model-online-badge {
        top: 14px !important;
        right: 14px !important;
        font-size: 13px;
        padding: 7px 12px;
    }

    .main-title {
        font-size: 31px;
    }

    .subtitle {
        font-size: 18px;
    }

    .section-title {
        font-size: 25px;
    }

    .prediction-option button {
        min-height: 125px !important;
        height: 125px !important;
        font-size: 18px !important;
    }

    .mobile-back-button {
        display: block;
        margin-bottom: 12px;
    }

    .mobile-back-button button {
        width: auto !important;
        min-width: 120px !important;
        min-height: 44px !important;
        font-size: 16px !important;
    }
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# MODEL CONFIGURATION
# =========================================================

MODEL_FILE = "churn_model.pkl"


MODEL_COLUMNS = [
    "SeniorCitizen",
    "gender",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]


# customerID is required for batch prediction.
# It is NEVER passed to the ML model.

BATCH_REQUIRED_COLUMNS = [
    "customerID"
] + MODEL_COLUMNS


CATEGORICAL_COLUMNS = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]


NUMERIC_COLUMNS = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]


ALLOWED_VALUES = {

    "SeniorCitizen": [
        0,
        1
    ],

    "gender": [
        "Female",
        "Male"
    ],

    "Partner": [
        "Yes",
        "No"
    ],

    "Dependents": [
        "Yes",
        "No"
    ],

    "PhoneService": [
        "Yes",
        "No"
    ],

    "MultipleLines": [
        "Yes",
        "No",
        "No phone service"
    ],

    "InternetService": [
        "DSL",
        "Fiber optic",
        "No"
    ],

    "OnlineSecurity": [
        "Yes",
        "No",
        "No internet service"
    ],

    "OnlineBackup": [
        "Yes",
        "No",
        "No internet service"
    ],

    "DeviceProtection": [
        "Yes",
        "No",
        "No internet service"
    ],

    "TechSupport": [
        "Yes",
        "No",
        "No internet service"
    ],

    "StreamingTV": [
        "Yes",
        "No",
        "No internet service"
    ],

    "StreamingMovies": [
        "Yes",
        "No",
        "No internet service"
    ],

    "Contract": [
        "Month-to-month",
        "One year",
        "Two year"
    ],

    "PaperlessBilling": [
        "Yes",
        "No"
    ],

    "PaymentMethod": [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ],
}


# =========================================================
# ONLY THE 16 MEANINGS YOU PROVIDED
# =========================================================

FIELD_HELP = {

    "Dependents":
        "Whether the customer has children or other people depending on them.",

    "tenure":
        "Number of months the customer has been using the service.",

    "PhoneService":
        "Whether the customer has phone service.",

    "MultipleLines":
        "Whether the customer has more than one phone line.",

    "InternetService":
        "The type of internet service the customer uses.",

    "OnlineSecurity":
        "Whether the customer has an online security service.",

    "OnlineBackup":
        "Whether the customer has an online backup service.",

    "DeviceProtection":
        "Whether the customer has protection for their device.",

    "TechSupport":
        "Whether the customer has technical support service.",

    "StreamingTV":
        "Whether the customer uses a TV streaming service.",

    "StreamingMovies":
        "Whether the customer uses a movie streaming service.",

    "Contract":
        "The type of service contract the customer has.",

    "PaperlessBilling":
        "Whether the customer receives bills electronically instead of on paper.",

    "PaymentMethod":
        "How the customer pays their bill.",

    "MonthlyCharges":
        "The amount the customer pays for the service each month.",

    "TotalCharges":
        "The total amount charged to the customer for the service so far."

}


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource(show_spinner=False)
def load_model():

    model_data = joblib.load(
        MODEL_FILE
    )

    preprocessor = model_data[
        "preprocessor"
    ]

    model = model_data[
        "model"
    ]

    return preprocessor, model


# =========================================================
# MODEL LOADING
# =========================================================

try:

    with st.spinner("Please wait..."):

        preprocessor, model = load_model()

except Exception as e:

    st.error(
        "❌ Unable to load the ML model."
    )

    st.error(
        f"Error: {e}"
    )

    st.stop()


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def normalize_input_columns(df):

    df = df.copy()

    df.columns = [
        str(column).strip()
        for column in df.columns
    ]

    return df


# =========================================================
# BATCH VALIDATION
# =========================================================

def validate_batch_data(df):

    errors = []

    warnings = []


    # -----------------------------------------------------
    # REQUIRED COLUMNS
    # -----------------------------------------------------

    missing = [

        column

        for column in BATCH_REQUIRED_COLUMNS

        if column not in df.columns

    ]


    if missing:

        errors.append(

            "Missing required columns: "
            + ", ".join(missing)

        )

        return errors, warnings


    # -----------------------------------------------------
    # NUMERIC COLUMNS
    # -----------------------------------------------------

    for column in NUMERIC_COLUMNS:

        converted = pd.to_numeric(
            df[column],
            errors="coerce"
        )


        invalid = (

            converted.isna()

            &

            df[column].notna()

        )


        if invalid.any():

            rows = (

                invalid[invalid]
                .index
                .tolist()

            )

            rows = [
                row + 2
                for row in rows[:10]
            ]


            errors.append(

                f"{column} contains non-numeric values. "
                f"Example CSV/Excel rows: {rows}"

            )


        df[column] = converted


    # -----------------------------------------------------
    # EMPTY NUMERIC VALUES
    # -----------------------------------------------------

    for column in NUMERIC_COLUMNS:

        if df[column].isna().any():

            rows = (

                df.index[
                    df[column].isna()
                ]
                .tolist()

            )

            rows = [
                row + 2
                for row in rows[:10]
            ]


            errors.append(

                f"{column} contains empty values. "
                f"Example rows: {rows}"

            )


    # -----------------------------------------------------
    # TENURE
    # -----------------------------------------------------

    if "tenure" in df.columns:

        bad = (

            (df["tenure"] < 0)

            |

            (df["tenure"] > 100)

        )


        if bad.any():

            errors.append(
                "Tenure must be between 0 and 100 months."
            )


    # -----------------------------------------------------
    # MONTHLY CHARGES
    # -----------------------------------------------------

    if "MonthlyCharges" in df.columns:

        if (
            df["MonthlyCharges"] < 0
        ).any():

            errors.append(
                "MonthlyCharges cannot contain negative values."
            )


    # -----------------------------------------------------
    # TOTAL CHARGES
    # -----------------------------------------------------

    if "TotalCharges" in df.columns:

        if (
            df["TotalCharges"] < 0
        ).any():

            errors.append(
                "TotalCharges cannot contain negative values."
            )


    # -----------------------------------------------------
    # CATEGORICAL VALUES
    # -----------------------------------------------------

    for column in CATEGORICAL_COLUMNS:

        if column not in df.columns:

            continue


        actual_values = set(

            df[column]
            .dropna()
            .astype(str)
            .unique()

        )


        allowed_values = set(
            ALLOWED_VALUES[column]
        )


        unexpected = sorted(

            actual_values
            -
            allowed_values

        )


        if unexpected:

            errors.append(

                f"{column} contains unsupported value(s): "
                f"{', '.join(unexpected)}. "
                f"Allowed values: "
                f"{', '.join(ALLOWED_VALUES[column])}"

            )


        if df[column].isna().any():

            errors.append(
                f"{column} contains empty values."
            )


    # -----------------------------------------------------
    # SENIOR CITIZEN
    # -----------------------------------------------------

    if "SeniorCitizen" in df.columns:

        bad = ~df[
            "SeniorCitizen"
        ].isin([0, 1])


        if bad.any():

            errors.append(
                "SeniorCitizen must contain only 0 or 1."
            )


    # -----------------------------------------------------
    # EXTRA COLUMNS
    # -----------------------------------------------------

    extra_columns = [

        column

        for column in df.columns

        if column not in BATCH_REQUIRED_COLUMNS

    ]


    if extra_columns:

        warnings.append(

            "Extra columns will be preserved in the output "
            "but ignored by the ML model: "

            +
            ", ".join(extra_columns)

        )


    return errors, warnings


# =========================================================
# CUSTOMER ID
# =========================================================

def get_customer_id_column(df):

    possible_names = [

        "customerID",

        "CustomerID",

        "customer_id",

        "Customer ID",

        "ID",

        "id"

    ]


    for column in possible_names:

        if column in df.columns:

            return column


    return None


# =========================================================
# RETENTION EXPLANATIONS & RECOMMENDATIONS
# =========================================================

# These explanations are transparent, rule-based business reasons.
# They do not claim to be SHAP/model-feature explanations.
def get_risk_reasons(row):
    reasons = []

    if row.get("Contract") == "Month-to-month":
        reasons.append("Month-to-month contract")

    if row.get("TechSupport") == "No" and row.get("InternetService") != "No":
        reasons.append("No technical support")

    if row.get("OnlineSecurity") == "No" and row.get("InternetService") != "No":
        reasons.append("No online security")

    if row.get("OnlineBackup") == "No" and row.get("InternetService") != "No":
        reasons.append("No online backup")

    if row.get("PaymentMethod") == "Electronic check":
        reasons.append("Electronic check payment")

    if pd.notna(row.get("MonthlyCharges")) and float(row["MonthlyCharges"]) >= 80:
        reasons.append("High monthly charges")

    if pd.notna(row.get("tenure")) and float(row["tenure"]) <= 12:
        reasons.append("Short customer tenure")

    if not reasons:
        reasons.append("No major rule-based retention risk factor identified")

    return reasons[:5]


def make_solution(row):
    actions = []

    if row["Contract"] == "Month-to-month":
        actions.append("Offer a longer-term contract option")

    if (
        row["TechSupport"] == "No"
        and row["InternetService"] != "No"
    ):
        actions.append("Offer technical support assistance")

    if (
        row["OnlineSecurity"] == "No"
        and row["InternetService"] != "No"
    ):
        actions.append("Offer online security support")

    if (
        row["OnlineBackup"] == "No"
        and row["InternetService"] != "No"
    ):
        actions.append("Offer online backup option")

    if row["PaymentMethod"] == "Electronic check":
        actions.append("Review payment options")

    if not actions:
        actions.append("Contact customer and review service satisfaction")

    return actions


def get_retention_priority(risk_level, is_high_value=False):
    if risk_level == "HIGH" and is_high_value:
        return "CRITICAL"
    if risk_level == "HIGH":
        return "HIGH"
    if risk_level == "MEDIUM" and is_high_value:
        return "HIGH"
    if risk_level == "MEDIUM":
        return "MEDIUM"
    return "LOW"


def get_risk_label(probability):
    if probability < 0.30:
        return "LOW"
    if probability < 0.70:
        return "MEDIUM"
    return "HIGH"


# =========================================================
# BATCH PREDICTION
# =========================================================

def predict_batch(df):
    model_input = df[MODEL_COLUMNS].copy()

    processed_data = preprocessor.transform(model_input)
    probabilities = model.predict_proba(processed_data)[:, 1]
    predictions = model.predict(processed_data)

    result = df.copy()

    result["Churn Prediction"] = [
        "Likely to Churn" if str(prediction) == "Yes" else "Likely to Stay"
        for prediction in predictions
    ]

    result["Churn Probability"] = (probabilities * 100).round(1)
    result["Risk Level"] = [get_risk_label(p) for p in probabilities]

    result["Risk Factors"] = result.apply(
        lambda row: "; ".join(get_risk_reasons(row)),
        axis=1
    )

    result["Recommended Action"] = result.apply(
        lambda row: "; ".join(make_solution(row)),
        axis=1
    )

    # High-value is defined relative to the uploaded batch:
    # customers in the top 25% of MonthlyCharges.
    value_threshold = result["MonthlyCharges"].quantile(0.75)
    result["High-Value Customer"] = result["MonthlyCharges"] >= value_threshold

    result["Retention Priority"] = result.apply(
        lambda row: get_retention_priority(
            row["Risk Level"],
            bool(row["High-Value Customer"])
        ),
        axis=1
    )

    return result


# =========================================================
# CREATE EXCEL
# =========================================================

def create_excel(result_df, churners_df):
    output = BytesIO()

    summary = pd.DataFrame({
        "Metric": [
            "Total Customers",
            "Likely To Churn",
            "Likely To Stay",
            "Churn Rate (%)",
            "High Risk",
            "Medium Risk",
            "Low Risk",
            "Critical Retention Priority"
        ],
        "Value": [
            len(result_df),
            len(churners_df),
            len(result_df) - len(churners_df),
            round((len(churners_df) / len(result_df)) * 100, 1) if len(result_df) else 0,
            int((result_df["Risk Level"] == "HIGH").sum()),
            int((result_df["Risk Level"] == "MEDIUM").sum()),
            int((result_df["Risk Level"] == "LOW").sum()),
            int((result_df["Retention Priority"] == "CRITICAL").sum())
        ]
    })

    priority_df = result_df[
        result_df["Retention Priority"].isin(["CRITICAL", "HIGH"])
    ].copy().sort_values("Churn Probability", ascending=False)

    high_value_df = result_df[
        (result_df["High-Value Customer"] == True)
        & (result_df["Churn Prediction"] == "Likely to Churn")
    ].copy().sort_values("Churn Probability", ascending=False)

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        result_df.to_excel(writer, index=False, sheet_name="All Predictions")
        churners_df.to_excel(writer, index=False, sheet_name="Likely To Churn")
        priority_df.to_excel(writer, index=False, sheet_name="Retention Priority")
        high_value_df.to_excel(writer, index=False, sheet_name="High-Value At Risk")
        summary.to_excel(writer, index=False, sheet_name="Summary")

        workbook = writer.book
        for worksheet in workbook.worksheets:
            worksheet.sheet_state = "visible"
        workbook.active = 0

    output.seek(0)
    return output.getvalue()

# =========================================================
# CREATE PDF
# =========================================================

def create_pdf(
    churners_df,
    total_customers,
    churn_count,
    churn_rate
):

    output = BytesIO()


    doc = SimpleDocTemplate(

        output,

        pagesize=landscape(A4),

        rightMargin=25,

        leftMargin=25,

        topMargin=25,

        bottomMargin=25

    )


    styles = getSampleStyleSheet()


    title_style = ParagraphStyle(

        "ReportTitle",

        parent=styles["Title"],

        alignment=TA_CENTER,

        fontSize=20,

        spaceAfter=12

    )


    body_style = ParagraphStyle(

        "Body",

        parent=styles["BodyText"],

        fontSize=9,

        leading=11

    )


    story = []


    story.append(

        Paragraph(
            "Customer Churn Prediction Report",
            title_style
        )

    )


    summary_data = [

        [
            "Total Customers",
            "Likely to Churn",
            "Likely to Stay",
            "Churn Rate"
        ],

        [

            str(total_customers),

            str(churn_count),

            str(
                total_customers
                -
                churn_count
            ),

            f"{churn_rate:.1f}%"

        ]

    ]


    summary_table = Table(

        summary_data,

        colWidths=[
            170,
            170,
            170,
            170
        ]

    )


    summary_table.setStyle(

        TableStyle([

            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.HexColor("#1f2937")
            ),

            (
                "TEXTCOLOR",
                (0, 0),
                (-1, 0),
                colors.white
            ),

            (
                "ALIGN",
                (0, 0),
                (-1, -1),
                "CENTER"
            ),

            (
                "FONTNAME",
                (0, 0),
                (-1, 0),
                "Helvetica-Bold"
            ),

            (
                "FONTNAME",
                (0, 1),
                (-1, 1),
                "Helvetica-Bold"
            ),

            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.grey
            ),

            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                8
            ),

            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                8
            )

        ])

    )


    story.append(
        summary_table
    )


    story.append(
        Spacer(1, 18)
    )


    story.append(

        Paragraph(
            "Customers Likely to Churn",
            styles["Heading2"]
        )

    )


    story.append(
        Spacer(1, 8)
    )


    if churners_df.empty:

        story.append(

            Paragraph(

                "No customers were classified as likely to churn.",

                body_style

            )

        )


    else:

        pdf_columns = [

            column

            for column in [

                get_customer_id_column(
                    churners_df
                ),

                "Churn Probability",

                "Risk Level",

                "Recommended Action"

            ]

            if (

                column is not None

                and

                column in churners_df.columns

            )

        ]


        if not any(

            column in pdf_columns

            for column in [

                "customerID",

                "CustomerID",

                "customer_id",

                "Customer ID",

                "ID",

                "id"

            ]

        ):

            temp = churners_df.copy()


            temp.insert(

                0,

                "Customer",

                [

                    f"Customer {i + 1}"

                    for i in range(
                        len(temp)
                    )

                ]

            )


            pdf_columns = [

                "Customer"

            ] + [

                column

                for column in pdf_columns

                if column in temp.columns

            ]


        else:

            temp = churners_df


        headers = pdf_columns


        data = [

            [

                Paragraph(
                    str(header),
                    body_style
                )

                for header in headers

            ]

        ]


        for _, row in temp[
            headers
        ].iterrows():

            data.append([

                Paragraph(

                    str(row[header]),

                    body_style

                )

                for header in headers

            ])


        widths = []


        for header in headers:

            if header == "Recommended Action":

                widths.append(360)

            elif header == "Churn Probability":

                widths.append(90)

            elif header == "Risk Level":

                widths.append(70)

            else:

                widths.append(100)


        table = Table(

            data,

            colWidths=widths,

            repeatRows=1

        )


        table.setStyle(

            TableStyle([

                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.HexColor("#1f2937")
                ),

                (
                    "TEXTCOLOR",
                    (0, 0),
                    (-1, 0),
                    colors.white
                ),

                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold"
                ),

                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.4,
                    colors.grey
                ),

                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "TOP"
                ),

                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    6
                ),

                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    6
                )

            ])

        )


        story.append(
            table
        )


    doc.build(
        story
    )


    output.seek(0)


    return output.getvalue()


# =========================================================
# ONLINE BADGE & SHARE-LINK HELPERS
# =========================================================
def render_model_online_badge():
    st.markdown(
        """<div class="model-online-badge"><span class="model-online-dot"></span>Model Online</div>""",
        unsafe_allow_html=True
    )


def encode_share_payload(payload):
    raw = json.dumps(payload, separators=(",", ":"), default=str).encode("utf-8")
    return base64.urlsafe_b64encode(zlib.compress(raw, 9)).decode("ascii")


SHARE_STORE_FILE = ".customer_retention_share_store.json"


def _read_share_store():
    try:
        if not os.path.exists(SHARE_STORE_FILE):
            return {}
        with open(SHARE_STORE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _write_share_store(store):
    temp_file = SHARE_STORE_FILE + ".tmp"
    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(store, f, separators=(",", ":"), default=str)
    os.replace(temp_file, SHARE_STORE_FILE)


def create_share_id(payload):
    share_id = uuid.uuid4().hex
    store = _read_share_store()
    store[share_id] = payload
    _write_share_store(store)
    return share_id


def load_share_id(share_id):
    return _read_share_store().get(share_id)


def decode_share_payload(encoded):
    try:
        raw = zlib.decompress(base64.urlsafe_b64decode(encoded.encode("ascii")))
        return json.loads(raw.decode("utf-8"))
    except Exception:
        return None


def render_copy_link_button(kind, payload, key, share_id=None, button_label="🔗 Share as Link"):
    """Render a copy-only share control. The actual URL is never displayed."""
    if share_id:
        parameter_name = "share_id"
        parameter_value = share_id
    else:
        parameter_name = "share"
        parameter_value = encode_share_payload(payload)

    value_js = json.dumps(parameter_value)
    parameter_js = json.dumps(parameter_name)
    button_js = json.dumps(button_label)

    html = f"""
    <div class="share-card">
      <button id="copy-{key}" style="width:100%;padding:12px 16px;border:0;border-radius:10px;background:#2563eb;color:white;font-size:16px;font-weight:700;cursor:pointer;">{button_label}</button>
      <div id="toast-{key}" style="position:fixed;left:50%;bottom:30px;transform:translateX(-50%) translateY(20px);background:#166534;color:white;padding:12px 20px;border-radius:10px;font-size:15px;font-weight:700;box-shadow:0 8px 24px rgba(0,0,0,.25);opacity:0;pointer-events:none;transition:all .25s ease;z-index:999999;">✓ Link copied to clipboard</div>
    </div>
    <script>
    const parameterName = {parameter_js};
    const parameterValue = {value_js};
    const kind = {json.dumps(kind)};
    const button = document.getElementById('copy-{key}');
    const toast = document.getElementById('toast-{key}');
    button.addEventListener('click', async () => {{
      try {{
        const current = new URL(window.parent.location.href);
        current.search = '';
        current.searchParams.set(parameterName, parameterValue);
        current.searchParams.set('view', kind);
        const shareUrl = current.toString();
        await navigator.clipboard.writeText(shareUrl);

        toast.style.opacity = '1';
        toast.style.transform = 'translateX(-50%) translateY(0)';
        button.textContent = '✓ Link Copied';

        setTimeout(() => {{
          toast.style.opacity = '0';
          toast.style.transform = 'translateX(-50%) translateY(20px)';
          button.textContent = {button_js};
        }}, 1800);
      }} catch (e) {{
        toast.textContent = 'Copy failed — allow clipboard access in your browser.';
        toast.style.background = '#991b1b';
        toast.style.opacity = '1';
        toast.style.transform = 'translateX(-50%) translateY(0)';
        setTimeout(() => {{
          toast.style.opacity = '0';
          toast.style.transform = 'translateX(-50%) translateY(20px)';
          toast.textContent = '✓ Link copied to clipboard';
          toast.style.background = '#166534';
        }}, 2200);
      }}
    }});
    </script>
    """
    components.html(html, height=80, scrolling=False)


def render_shared_single(payload):
    render_model_online_badge()
    st.markdown("<div class='section-title'>🔗 Shared Customer Prediction</div>", unsafe_allow_html=True)
    st.caption("This result was opened from a shared prediction link.")

    churn_probability = float(payload.get("churn_probability", 0))
    risk_level = payload.get("risk_level", "UNKNOWN")
    prediction_text = payload.get("prediction_text", "Unknown")
    retention_priority = payload.get("retention_priority", "LOW")
    risk_message = payload.get("risk_message", "")
    reasons = payload.get("reasons", [])
    actions = payload.get("actions", [])

    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Churn Probability", f"{churn_probability * 100:.1f}%")
    with col2:
        if risk_level == "HIGH": st.error(f"🔴 {risk_level} RISK")
        elif risk_level == "MEDIUM": st.warning(f"🟠 {risk_level} RISK")
        else: st.success(f"🟢 {risk_level} RISK")
    with col3: st.metric("Prediction", prediction_text)
    with col4:
        icon = {"CRITICAL":"🔴", "HIGH":"🟠", "MEDIUM":"🟡", "LOW":"🟢"}.get(retention_priority, "⚪")
        st.metric("Retention Priority", f"{icon} {retention_priority}")

    if risk_message: st.info(risk_message)
    st.markdown("### 🔎 Why this customer is at risk")
    for reason in reasons: st.write(f"• {reason}")
    st.markdown("### 🎯 Recommended Retention Actions")
    for number, action in enumerate(actions, start=1): st.write(f"**{number}.** {action}")


def render_shared_batch(payload):
    render_model_online_badge()
    st.markdown("<div class='section-title'>🔗 Shared Batch Prediction</div>", unsafe_allow_html=True)
    st.caption("This result was opened from a shared batch prediction link.")

    result_df = pd.DataFrame(payload.get("result", []))
    if result_df.empty:
        st.warning("The shared batch result contains no rows.")
        return

    churners_df = result_df[result_df.get("Churn Prediction", pd.Series(dtype=str)) == "Likely to Churn"].copy()
    total_customers = len(result_df)
    churn_count = len(churners_df)
    churn_rate = (churn_count / total_customers) * 100 if total_customers else 0
    high_risk_count = int((result_df["Risk Level"] == "HIGH").sum()) if "Risk Level" in result_df else 0

    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Total Customers", f"{total_customers:,}")
    with col2: st.metric("Likely to Churn", f"{churn_count:,}")
    with col3: st.metric("High Risk", f"{high_risk_count:,}")
    with col4: st.metric("Churn Rate", f"{churn_rate:.1f}%")

    customer_id_column = get_customer_id_column(result_df)
    display_columns = [customer_id_column, "Churn Probability", "Churn Prediction", "Risk Level", "High-Value Customer", "Retention Priority", "Risk Factors", "Recommended Action"]
    display_columns = [c for c in display_columns if c is not None and c in result_df.columns]
    unified = result_df[display_columns].copy()
    if "High-Value Customer" in unified.columns:
        unified["High-Value Customer"] = unified["High-Value Customer"].map({True:"Yes", False:"No"})
    st.markdown("### 👥 Customer Retention Analysis")
    st.dataframe(unified, use_container_width=True, hide_index=True)




def load_shared_result():
    view = st.query_params.get("view")
    if view not in {"single", "batch"}:
        return None, None

    share_id = st.query_params.get("share_id")
    if share_id:
        return view, load_share_id(share_id)

    # Backward compatibility for links created by the previous version.
    encoded = st.query_params.get("share")
    if not encoded:
        return None, None
    return view, decode_share_payload(encoded)


# =========================================================
# TITLE
# =========================================================

st.markdown(

    """
    <div class="main-title">
        📊 Customer Retention Intelligence System
    </div>
    """,

    unsafe_allow_html=True

)


st.markdown(

    """
    <div class="subtitle">
        Predict customer churn risk and identify customers
        who may need retention support.
    </div>
    """,

    unsafe_allow_html=True

)


# =========================================================
# SINGLE-PAGE NAVIGATION
# =========================================================
if "app_page" not in st.session_state:
    st.session_state.app_page = "prediction"
if "prediction_mode" not in st.session_state:
    st.session_state.prediction_mode = None
if "nav_open" not in st.session_state:
    st.session_state.nav_open = False

shared_view, shared_payload = load_shared_result()

def go_prediction_center():
    st.session_state.app_page = "prediction"
    st.session_state.prediction_mode = None
    st.session_state.nav_open = False

def go_about_model():
    st.session_state.app_page = "about"
    st.session_state.prediction_mode = None
    st.session_state.nav_open = False

# ---------------------------------------------------------
# HAMBURGER / NAVIGATION
# ---------------------------------------------------------
def open_navigation():
    st.session_state.nav_open = True

def close_navigation():
    st.session_state.nav_open = False

if not st.session_state.nav_open:
    # OPEN button
    st.markdown('<div class="nav-hamburger-wrap">', unsafe_allow_html=True)
    st.button(
        "☰",
        key="hamburger_open",
        help=None,
        on_click=open_navigation
    )
    st.markdown('</div>', unsafe_allow_html=True)

else:
    # Reserve the left side for the open navigation panel.
    st.markdown(
        '''
        <style>
        [data-testid="stMainBlockContainer"] {
            margin-left: 290px !important;
            width: calc(100% - 290px) !important;
            transition: margin-left 0.25s ease-in-out,
                        width 0.25s ease-in-out !important;
        }

        [data-testid="stMainBlockContainer"] > div {
            max-width: none !important;
            overflow: visible !important;
        }
        </style>
        ''',
        unsafe_allow_html=True
    )


    # The CLOSE button is physically INSIDE the open panel.
    with st.container(key="nav_panel"):
        st.button(
            "☰",
            key="hamburger_close",
            on_click=close_navigation
        )

        st.button(
            "Prediction Center",
            key="nav_prediction",
            on_click=go_prediction_center,
            use_container_width=True
        )

        st.button(
            "About Model",
            key="nav_about",
            on_click=go_about_model,
            use_container_width=True
        )

if shared_view and shared_payload is not None:
    st.session_state.app_page = "prediction"
    st.session_state.prediction_mode = shared_view
    if shared_view == "single":
        render_shared_single(shared_payload)
    else:
        render_shared_batch(shared_payload)
    st.stop()

if st.session_state.app_page == "about":
    st.markdown('<div class="section-title">📊 About Model</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Logistic Regression model performance</div>', unsafe_allow_html=True)

    # Four SVG speedometers animate clockwise from 0 to the exact
    # Logistic Regression performance values.
    gauge_html = r'''<!DOCTYPE html>
<html>
<head>
<style>
*{box-sizing:border-box}
body{margin:0;background:transparent;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:#f8fafc}
.gauges{width:100%;display:grid;grid-template-columns:repeat(4,minmax(220px,1fr));gap:18px;padding:12px 0 18px}
.gauge-card{min-height:335px;border:1px solid #334155;border-radius:18px;background:#1f2937;display:flex;flex-direction:column;align-items:center;padding:28px 18px 18px;box-shadow:0 8px 20px rgba(0,0,0,.14)}
.gauge-title{font-size:19px;font-weight:700;margin-bottom:7px;text-align:center}
.gauge-wrap{width:100%;max-width:330px;height:205px}
svg{width:100%;height:100%;overflow:visible}
.gauge-track{fill:none;stroke:#3b4658;stroke-width:22;stroke-linecap:round}
.gauge-progress{fill:none;stroke:#60a5fa;stroke-width:22;stroke-linecap:round}
.needle{stroke:#f8fafc;stroke-width:5;stroke-linecap:round}
.needle-center{fill:#f8fafc}
.scale-label{fill:#94a3b8;font-size:13px}
.value{margin-top:-1px;font-size:31px;font-weight:800;letter-spacing:.2px}
.meaning{color:#94a3b8;font-size:12px;margin-top:8px;text-align:center;line-height:1.35;min-height:34px;max-width:250px}
@media(max-width:1100px){.gauges{grid-template-columns:repeat(2,minmax(240px,1fr))}}
@media(max-width:620px){.gauges{grid-template-columns:1fr}.gauge-card{min-height:320px}}
</style>
</head>
<body>
<div class="gauges">
    <div class="gauge-card">
        <div class="gauge-title">Accuracy</div>
        <div class="gauge-wrap">
            <svg viewBox="0 0 320 200" aria-label="Accuracy 80.55 percent">
                <path class="gauge-track" d="M40 160 A120 120 0 0 1 280 160"/>
                <path id="arc-accuracy" class="gauge-progress" d="M40 160 A120 120 0 0 1 280 160"/>
                <line id="needle-accuracy" class="needle" x1="160" y1="160" x2="252" y2="160" transform="rotate(180 160 160)"/>
                <circle class="needle-center" cx="160" cy="160" r="7"/>
                <text class="scale-label" x="38" y="184">0</text>
                <text class="scale-label" x="267" y="184">100</text>
            </svg>
        </div>
        <div id="value-accuracy" class="value">0.00%</div>
        <div class="meaning">Correct predictions out of all predictions</div>
    </div>

    <div class="gauge-card">
        <div class="gauge-title">ROC-AUC</div>
        <div class="gauge-wrap">
            <svg viewBox="0 0 320 200" aria-label="ROC-AUC 84.21 percent">
                <path class="gauge-track" d="M40 160 A120 120 0 0 1 280 160"/>
                <path id="arc-roc" class="gauge-progress" d="M40 160 A120 120 0 0 1 280 160"/>
                <line id="needle-roc" class="needle" x1="160" y1="160" x2="252" y2="160" transform="rotate(180 160 160)"/>
                <circle class="needle-center" cx="160" cy="160" r="7"/>
                <text class="scale-label" x="38" y="184">0</text>
                <text class="scale-label" x="267" y="184">100</text>
            </svg>
        </div>
        <div id="value-roc" class="value">0.00%</div>
        <div class="meaning">Ability to distinguish churners from non-churners</div>
    </div>

    <div class="gauge-card">
        <div class="gauge-title">Recall</div>
        <div class="gauge-wrap">
            <svg viewBox="0 0 320 200" aria-label="Recall 56.0 percent">
                <path class="gauge-track" d="M40 160 A120 120 0 0 1 280 160"/>
                <path id="arc-recall" class="gauge-progress" d="M40 160 A120 120 0 0 1 280 160"/>
                <line id="needle-recall" class="needle" x1="160" y1="160" x2="252" y2="160" transform="rotate(180 160 160)"/>
                <circle class="needle-center" cx="160" cy="160" r="7"/>
                <text class="scale-label" x="38" y="184">0</text>
                <text class="scale-label" x="267" y="184">100</text>
            </svg>
        </div>
        <div id="value-recall" class="value">0.00%</div>
        <div class="meaning">Churners correctly identified by the model</div>
    </div>

    <div class="gauge-card">
        <div class="gauge-title">F1-Score</div>
        <div class="gauge-wrap">
            <svg viewBox="0 0 320 200" aria-label="F1-Score 60.0 percent">
                <path class="gauge-track" d="M40 160 A120 120 0 0 1 280 160"/>
                <path id="arc-f1" class="gauge-progress" d="M40 160 A120 120 0 0 1 280 160"/>
                <line id="needle-f1" class="needle" x1="160" y1="160" x2="252" y2="160" transform="rotate(180 160 160)"/>
                <circle class="needle-center" cx="160" cy="160" r="7"/>
                <text class="scale-label" x="38" y="184">0</text>
                <text class="scale-label" x="267" y="184">100</text>
            </svg>
        </div>
        <div id="value-f1" class="value">0.00%</div>
        <div class="meaning">Balance between precision and recall</div>
    </div>
</div>

<script>
(function(){
    const gauges=[
        {key:"accuracy",target:80.55},
        {key:"roc",target:84.21},
        {key:"recall",target:56},
        {key:"f1",target:60}
    ];

    const arcLength=Math.PI*120;
    const duration=1800;

    // Every gauge starts at exactly 0%: needle at the far-left end
    // and progress arc completely hidden.
    gauges.forEach(g=>{
        const arc=document.getElementById("arc-"+g.key);
        const needle=document.getElementById("needle-"+g.key);
        arc.style.strokeDasharray=arcLength+" "+arcLength;
        arc.style.strokeDashoffset=arcLength;
        needle.setAttribute("transform","rotate(180 160 160)");
    });

    const start=performance.now();

    function ease(t){
        return 1-Math.pow(1-t,3);
    }

    function animate(now){
        const progress=Math.min((now-start)/duration,1);
        const eased=ease(progress);

        gauges.forEach(g=>{
            const value=g.target*eased;
            const angle=180+(value*1.8);

            // IMPORTANT: 180deg = 0%, then the needle moves CLOCKWISE
            // through the top of the gauge until it reaches the target.
            document.getElementById("needle-"+g.key)
                .setAttribute("transform","rotate("+angle+" 160 160)");

            document.getElementById("arc-"+g.key).style.strokeDashoffset=
                arcLength*(1-value/100);

            document.getElementById("value-"+g.key).textContent=
                value.toFixed(2)+"%";
        });

        if(progress<1){
            requestAnimationFrame(animate);
        }else{
            // Force the final frame to the EXACT metric values.
            gauges.forEach(g=>{
                const finalAngle=180+(g.target*1.8);
                document.getElementById("needle-"+g.key)
                    .setAttribute("transform","rotate("+finalAngle+" 160 160)");
                document.getElementById("arc-"+g.key).style.strokeDashoffset=
                    arcLength*(1-g.target/100);
                document.getElementById("value-"+g.key).textContent=
                    g.target.toFixed(2)+"%";
            });
        }
    }

    requestAnimationFrame(animate);
})();
</script>
</body>
</html>'''
    components.html(gauge_html, height=365, scrolling=False)

    st.markdown("### 🏆 Logistic Regression")
    st.table(pd.DataFrame([{
        "Model": "🏆 Logistic Regression",
        "Accuracy": "80.55%",
        "ROC-AUC": "84.21%",
        "Recall": "56%",
        "F1-Score": "60%"
    }]))
    st.stop()


# =========================================================
# PREDICTION MODE
# =========================================================

# =========================================================
# INITIAL SCREEN
# =========================================================

if st.session_state.prediction_mode is None:

    st.markdown(

        """
        <div class="section-title">
            Choose Prediction Type
        </div>
        """,

        unsafe_allow_html=True

    )


    render_model_online_badge()

    st.markdown(

        """
        <div style="
            font-size:18px;
            color:#9ca3af;
            margin-bottom:25px;
        ">
            Select how you want to predict customer churn.
        </div>
        """,

        unsafe_allow_html=True

    )


    option_col1, option_col2 = st.columns(
        2,
        gap="large"
    )


    # -----------------------------------------------------
    # SINGLE CUSTOMER
    # -----------------------------------------------------

    with option_col1:

        st.markdown(
            '<div class="prediction-option">',
            unsafe_allow_html=True
        )


        if st.button(

            "👤 Single Customer\n\n"
            "Predict one customer at a time",

            key="single_customer_option",

            use_container_width=True

        ):

            st.session_state.app_page = "prediction"
            st.session_state.prediction_mode = "single"
            st.rerun()


        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )


    # -----------------------------------------------------
    # BATCH PREDICTION
    # -----------------------------------------------------

    with option_col2:

        st.markdown(
            '<div class="prediction-option">',
            unsafe_allow_html=True
        )


        if st.button(

            "📂 Batch Prediction\n\n"
            "Predict multiple customers from a file",

            key="batch_prediction_option",

            use_container_width=True

        ):

            st.session_state.app_page = "prediction"
            st.session_state.prediction_mode = "batch"
            st.rerun()


        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )


# =========================================================
# SINGLE CUSTOMER MODE
# =========================================================

elif st.session_state.prediction_mode == "single":

    render_model_online_badge()

    # =====================================================
    # MOBILE BACK BUTTON
    # =====================================================
    # Hidden on laptop/desktop. On mobile it returns to the
    # prediction-type screen without relying on the browser
    # navigation controls.

    st.markdown(
        '<div class="mobile-back-button">',
        unsafe_allow_html=True
    )

    if st.button(
        "← Back",
        key="mobile_back_single"
    ):
        st.session_state.app_page = "prediction"
        st.session_state.prediction_mode = None
        st.rerun()

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )


    # =====================================================
    # CUSTOMER INFORMATION
    # =====================================================

    st.markdown(

        """
        <div class="section-title">
            👤 Customer Information
        </div>
        """,

        unsafe_allow_html=True

    )


    col1, col2, col3 = st.columns(3)


    # -----------------------------------------------------
    # SENIOR CITIZEN / GENDER
    # NO HELP ICONS
    # -----------------------------------------------------

    with col1:

        senior_citizen = st.selectbox(

            "Senior Citizen",

            [0, 1],

            index=None,

            placeholder="Select an option",

            format_func=lambda x:
                "Yes" if x == 1 else "No"

        )


        gender = st.selectbox(

            "Gender",

            [
                "Female",
                "Male"
            ],

            index=None,

            placeholder="Select an option"

        )


    # -----------------------------------------------------
    # PARTNER / DEPENDENTS
    # -----------------------------------------------------

    with col2:

        partner = st.selectbox(

            "Partner",

            [
                "Yes",
                "No"
            ],

            index=None,

            placeholder="Select an option"

        )


        dependents = st.selectbox(

            "Dependents",

            [
                "Yes",
                "No"
            ],

            index=None,

            placeholder="Select an option",

            help=FIELD_HELP[
                "Dependents"
            ]

        )


    # =====================================================
    # PHONE & INTERNET
    # =====================================================

    st.markdown(

        """
        <div class="section-title">
            📱 Phone & Internet Services
        </div>
        """,

        unsafe_allow_html=True

    )


    col1, col2, col3 = st.columns(3)


    with col1:

        phone_service = st.selectbox(

            "Phone Service",

            [
                "Yes",
                "No"
            ],

            index=None,

            placeholder="Select an option",

            help=FIELD_HELP[
                "PhoneService"
            ]

        )


        multiple_lines = st.selectbox(

            "Multiple Lines",

            [
                "Yes",
                "No",
                "No phone service"
            ],

            index=None,

            placeholder="Select an option",

            help=FIELD_HELP[
                "MultipleLines"
            ]

        )


        internet_service = st.selectbox(

            "Internet Service",

            [
                "DSL",
                "Fiber optic",
                "No"
            ],

            index=None,

            placeholder="Select an option",

            help=FIELD_HELP[
                "InternetService"
            ]

        )


    with col2:

        online_security = st.selectbox(

            "Online Security",

            [
                "Yes",
                "No",
                "No internet service"
            ],

            index=None,

            placeholder="Select an option",

            help=FIELD_HELP[
                "OnlineSecurity"
            ]

        )


        online_backup = st.selectbox(

            "Online Backup",

            [
                "Yes",
                "No",
                "No internet service"
            ],

            index=None,

            placeholder="Select an option",

            help=FIELD_HELP[
                "OnlineBackup"
            ]

        )


        device_protection = st.selectbox(

            "Device Protection",

            [
                "Yes",
                "No",
                "No internet service"
            ],

            index=None,

            placeholder="Select an option",

            help=FIELD_HELP[
                "DeviceProtection"
            ]

        )


    with col3:

        tech_support = st.selectbox(

            "Tech Support",

            [
                "Yes",
                "No",
                "No internet service"
            ],

            index=None,

            placeholder="Select an option",

            help=FIELD_HELP[
                "TechSupport"
            ]

        )


    # =====================================================
    # STREAMING
    # =====================================================

    st.markdown(

        """
        <div class="section-title">
            📺 Streaming Services
        </div>
        """,

        unsafe_allow_html=True

    )


    col1, col2, col3 = st.columns(3)


    with col1:

        streaming_tv = st.selectbox(

            "Streaming TV",

            [
                "Yes",
                "No",
                "No internet service"
            ],

            index=None,

            placeholder="Select an option",

            help=FIELD_HELP[
                "StreamingTV"
            ]

        )


    with col2:

        streaming_movies = st.selectbox(

            "Streaming Movies",

            [
                "Yes",
                "No",
                "No internet service"
            ],

            index=None,

            placeholder="Select an option",

            help=FIELD_HELP[
                "StreamingMovies"
            ]

        )


    # =====================================================
    # BILLING & CONTRACT
    # =====================================================

    st.markdown(

        """
        <div class="section-title">
            💳 Billing & Contract Information
        </div>
        """,

        unsafe_allow_html=True

    )


    col1, col2, col3 = st.columns(3)


    with col1:

        contract = st.selectbox(

            "Contract",

            [
                "Month-to-month",
                "One year",
                "Two year"
            ],

            index=None,

            placeholder="Select an option",

            help=FIELD_HELP[
                "Contract"
            ]

        )


        paperless_billing = st.selectbox(

            "Paperless Billing",

            [
                "Yes",
                "No"
            ],

            index=None,

            placeholder="Select an option",

            help=FIELD_HELP[
                "PaperlessBilling"
            ]

        )


    with col2:

        payment_method = st.selectbox(

            "Payment Method",

            [

                "Electronic check",

                "Mailed check",

                "Bank transfer (automatic)",

                "Credit card (automatic)"

            ],

            index=None,

            placeholder="Select an option",

            help=FIELD_HELP[
                "PaymentMethod"
            ]

        )


        tenure_input = st.text_input(

            "Tenure",

            value="",

            placeholder="Enter number of months",

            help=FIELD_HELP[
                "tenure"
            ]

        )


    with col3:

        monthly_charges_input = st.text_input(

            "Monthly Charges",

            value="",

            placeholder="Enter monthly amount",

            help=FIELD_HELP[
                "MonthlyCharges"
            ]

        )


        total_charges_input = st.text_input(

            "Total Charges",

            value="",

            placeholder="Enter total amount",

            help=FIELD_HELP[
                "TotalCharges"
            ]

        )


    # =====================================================
    # PREDICTION BUTTON
    # =====================================================

    st.markdown("---")


    predict_button = st.button(

        "🔍 Predict Churn Risk",

        use_container_width=True

    )


    if predict_button:

        # -------------------------------------------------
        # REQUIRED INPUT CHECK
        # -------------------------------------------------

        required_inputs = {

            "Senior Citizen":
                senior_citizen,

            "Gender":
                gender,

            "Partner":
                partner,

            "Dependents":
                dependents,

            "Phone Service":
                phone_service,

            "Multiple Lines":
                multiple_lines,

            "Internet Service":
                internet_service,

            "Online Security":
                online_security,

            "Online Backup":
                online_backup,

            "Device Protection":
                device_protection,

            "Tech Support":
                tech_support,

            "Streaming TV":
                streaming_tv,

            "Streaming Movies":
                streaming_movies,

            "Contract":
                contract,

            "Paperless Billing":
                paperless_billing,

            "Payment Method":
                payment_method,

            "Tenure":
                tenure_input,

            "Monthly Charges":
                monthly_charges_input,

            "Total Charges":
                total_charges_input

        }


        missing_fields = []


        for field_name, value in required_inputs.items():

            if (

                value is None

                or

                str(value).strip() == ""

            ):

                missing_fields.append(
                    field_name
                )


        if missing_fields:

            st.warning(

                "⚠️ Please complete all fields "
                "before making a prediction."

            )


            st.write(

                "**Missing fields:** "
                +
                ", ".join(missing_fields)

            )


            st.stop()


        # -------------------------------------------------
        # NUMERIC CONVERSION
        # -------------------------------------------------

        try:

            tenure = float(
                tenure_input
            )

            monthly_charges = float(
                monthly_charges_input
            )

            total_charges = float(
                total_charges_input
            )

        except ValueError:

            st.error(

                "❌ Please enter valid numbers for "
                "Tenure, Monthly Charges, and Total Charges."

            )

            st.stop()


        # -------------------------------------------------
        # RANGE CHECKS
        # -------------------------------------------------

        if tenure < 0 or tenure > 100:

            st.error(
                "❌ Tenure must be between 0 and 100 months."
            )

            st.stop()


        if monthly_charges < 0:

            st.error(
                "❌ Monthly Charges cannot be negative."
            )

            st.stop()


        if total_charges < 0:

            st.error(
                "❌ Total Charges cannot be negative."
            )

            st.stop()


        # -------------------------------------------------
        # CREATE DATAFRAME
        # -------------------------------------------------

        customer_data = pd.DataFrame([{

            "SeniorCitizen":
                senior_citizen,

            "gender":
                gender,

            "Partner":
                partner,

            "Dependents":
                dependents,

            "tenure":
                tenure,

            "PhoneService":
                phone_service,

            "MultipleLines":
                multiple_lines,

            "InternetService":
                internet_service,

            "OnlineSecurity":
                online_security,

            "OnlineBackup":
                online_backup,

            "DeviceProtection":
                device_protection,

            "TechSupport":
                tech_support,

            "StreamingTV":
                streaming_tv,

            "StreamingMovies":
                streaming_movies,

            "Contract":
                contract,

            "PaperlessBilling":
                paperless_billing,

            "PaymentMethod":
                payment_method,

            "MonthlyCharges":
                monthly_charges,

            "TotalCharges":
                total_charges

        }])


        # -------------------------------------------------
        # PREDICT
        # -------------------------------------------------

        try:

            with st.spinner(
                "Predicting customer churn..."
            ):

                processed_data = preprocessor.transform(
                    customer_data
                )

                churn_probability = model.predict_proba(
                    processed_data
                )[0][1]

                churn_prediction = model.predict(
                    processed_data
                )[0]

        except Exception as e:

            st.error(
                "❌ Prediction failed."
            )

            st.error(
                f"Error: {e}"
            )

            st.stop()


        # -------------------------------------------------
        # RISK
        # -------------------------------------------------
        risk_level = get_risk_label(churn_probability)

        if risk_level == "LOW":
            risk_message = "This customer currently has a low churn risk."
        elif risk_level == "MEDIUM":
            risk_message = "This customer has a moderate churn risk."
        else:
            risk_message = (
                "This customer has a high churn risk and may require "
                "retention attention."
            )

        # -------------------------------------------------
        # RULE-BASED RISK FACTORS & ACTIONS
        # -------------------------------------------------
        single_reasons = get_risk_reasons(customer_data.iloc[0])
        single_actions = make_solution(customer_data.iloc[0])
        retention_priority = get_retention_priority(risk_level, False)

        # -------------------------------------------------
        # RESULT
        # -------------------------------------------------
        st.markdown(
            """
            <div class="section-title">
                📈 Churn Prediction
            </div>
            """,
            unsafe_allow_html=True
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Churn Probability",
                f"{churn_probability * 100:.1f}%"
            )

        with col2:
            if risk_level == "HIGH":
                st.error(f"🔴 {risk_level} RISK")
            elif risk_level == "MEDIUM":
                st.warning(f"🟠 {risk_level} RISK")
            else:
                st.success(f"🟢 {risk_level} RISK")

        with col3:
            prediction_text = (
                "Likely to Churn"
                if churn_prediction == "Yes"
                else "Likely to Stay"
            )
            st.metric("Prediction", prediction_text)

        with col4:
            priority_icon = {
                "CRITICAL": "🔴",
                "HIGH": "🟠",
                "MEDIUM": "🟡",
                "LOW": "🟢"
            }[retention_priority]
            st.metric(
                "Retention Priority",
                f"{priority_icon} {retention_priority}"
            )

        st.info(risk_message)

        st.markdown(
            "### 🔎 Why this customer is at risk"
        )
        for reason in single_reasons:
            st.write(f"• {reason}")

        st.markdown(
            "### 🎯 Recommended Retention Actions"
        )
        for number, action in enumerate(single_actions, start=1):
            st.write(f"**{number}.** {action}")

        st.caption(
            "Risk factors and retention actions shown above are transparent, "
            "rule-based business explanations and are not direct model-feature explanations."
        )

        # =================================================
        # SHARE SINGLE RESULT
        # =================================================
        single_share_payload = {
            "churn_probability": float(churn_probability),
            "risk_level": risk_level,
            "prediction_text": prediction_text,
            "retention_priority": retention_priority,
            "risk_message": risk_message,
            "reasons": single_reasons,
            "actions": single_actions,
        }
        if st.button("🔗 Share as Link", key="create_single_share", use_container_width=True):
            st.query_params["share"] = encode_share_payload(single_share_payload)
            st.query_params["view"] = "single"
            st.session_state.single_share_ready = True
            st.rerun()

        if st.session_state.get("single_share_ready") and st.query_params.get("share"):
            render_copy_link_button(
                "single",
                single_share_payload,
                "single-share"
            )


# =========================================================
# BATCH PREDICTION MODE
# =========================================================

elif st.session_state.prediction_mode == "batch":

    render_model_online_badge()

    # =====================================================
    # MOBILE BACK BUTTON
    # =====================================================
    # Hidden on laptop/desktop. On mobile it returns to the
    # prediction-type screen without relying on the browser
    # navigation controls.

    st.markdown(
        '<div class="mobile-back-button">',
        unsafe_allow_html=True
    )

    if st.button("← Back", key="mobile_back_batch"):
        st.session_state.app_page = "prediction"
        st.session_state.prediction_mode = None
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    # =====================================================
    # TITLE
    # =====================================================
    st.markdown(
        """
        <div class="section-title">
            📂 Batch Customer Prediction
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write(
        "Upload a CSV, Excel (.xlsx), or JSON file containing multiple "
        "customers. The uploaded data must include Customer ID and the "
        "following 19 features used for prediction."
    )

    # =====================================================
    # REQUIRED COLUMNS & FIELD MEANINGS
    # =====================================================
    with st.expander("📋 Required columns & field meanings"):
        st.write("**Required columns:**")

        # Only the fields that already have meanings in FIELD_HELP
        # receive the small information icon.
        required_fields_html = ""

        for i in range(0, len(BATCH_REQUIRED_COLUMNS), 3):
            row = BATCH_REQUIRED_COLUMNS[i:i + 3]

            required_fields_html += '<div class="required-fields-row">'

            for field in row:
                if field in FIELD_HELP:
                    meaning = (
                        FIELD_HELP[field]
                        .replace("&", "&amp;")
                        .replace('"', "&quot;")
                        .replace("<", "&lt;")
                        .replace(">", "&gt;")
                    )

                    field_html = (
                        f'<span class="required-field-content">'
                        f'{field}'
                        f'<span class="required-field-info" '
                        f'title="{meaning}">ⓘ</span>'
                        f'</span>'
                    )
                else:
                    field_html = field

                required_fields_html += (
                    f'<span class="required-field">'
                    f'• {field_html}'
                    f'</span>'
                )

            required_fields_html += '</div>'

        st.markdown(
            f'<div class="required-fields-list">'
            f'{required_fields_html}'
            f'</div>',
            unsafe_allow_html=True
        )

        st.caption(
            "Hover over the ⓘ symbol beside a field to see its meaning."
        )

    # =====================================================
    # FILE UPLOAD
    # =====================================================
    uploaded_file = st.file_uploader(
        "Upload file",
        type=["csv", "xlsx", "json"],
        help="Supported input formats: CSV, XLSX, JSON",
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        try:
            extension = uploaded_file.name.lower().split(".")[-1]

            if extension == "csv":
                batch_df = pd.read_csv(uploaded_file)
            elif extension == "xlsx":
                batch_df = pd.read_excel(uploaded_file)
            elif extension == "json":
                batch_df = pd.read_json(uploaded_file)
            else:
                st.error("Unsupported file type.")
                st.stop()

            batch_df = normalize_input_columns(batch_df)

            if batch_df.empty:
                st.error("❌ The uploaded file contains no customer rows.")
                st.stop()

            st.success(
                f"Loaded **{len(batch_df):,} customer(s)** from `{uploaded_file.name}`."
            )

            st.markdown(
                "<div class='section-title'>👀 Data Preview</div>",
                unsafe_allow_html=True
            )
            st.dataframe(batch_df.head(10), use_container_width=True)

            errors, warnings = validate_batch_data(batch_df)

            for warning in warnings:
                st.warning(f"⚠️ {warning}")

            if errors:
                st.error("❌ The file cannot be processed yet.")
                for error in errors:
                    st.write(f"- {error}")
                st.stop()

            st.success("✅ File validation passed.")

            # =================================================
            # PERSISTENT RESULTS
            # =================================================
            if "batch_result_df" not in st.session_state:
                st.session_state.batch_result_df = None
                st.session_state.batch_churners_df = None
                st.session_state.batch_source_name = None

            if st.session_state.batch_source_name != uploaded_file.name:
                st.session_state.batch_result_df = None
                st.session_state.batch_churners_df = None
                st.session_state.batch_source_name = uploaded_file.name

            if st.button(
                "🚀 Run Batch Churn Prediction",
                use_container_width=True
            ):
                with st.spinner("Running churn predictions..."):
                    prediction_result = predict_batch(batch_df)

                st.session_state.batch_result_df = prediction_result
                st.session_state.batch_churners_df = prediction_result[
                    prediction_result["Churn Prediction"] == "Likely to Churn"
                ].copy()

            # =================================================
            # SHOW RESULTS
            # =================================================
            if st.session_state.batch_result_df is not None:
                result_df = st.session_state.batch_result_df
                churners_df = st.session_state.batch_churners_df

                total_customers = len(result_df)
                churn_count = len(churners_df)
                stay_count = total_customers - churn_count
                churn_rate = (
                    (churn_count / total_customers) * 100
                    if total_customers else 0
                )

                high_risk_count = int((result_df["Risk Level"] == "HIGH").sum())
                medium_risk_count = int((result_df["Risk Level"] == "MEDIUM").sum())
                low_risk_count = int((result_df["Risk Level"] == "LOW").sum())
                critical_count = int(
                    (result_df["Retention Priority"] == "CRITICAL").sum()
                )

                st.markdown("---")
                st.markdown(
                    "<div class='section-title'>📊 Batch Prediction Summary</div>",
                    unsafe_allow_html=True
                )

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Customers", f"{total_customers:,}")
                with col2:
                    st.metric("Likely to Churn", f"{churn_count:,}")
                with col3:
                    st.metric("High Risk", f"{high_risk_count:,}")
                with col4:
                    st.metric("Churn Rate", f"{churn_rate:.1f}%")

                # =================================================
                # SINGLE UNIFIED CUSTOMER OUTPUT
                # =================================================
                # There is intentionally ONE customer result table.
                # High-value status and retention priority are columns
                # in this same table, not separate outputs.
                st.markdown("### 👥 Customer Retention Analysis")

                customer_id_column = get_customer_id_column(result_df)

                display_columns = [
                    customer_id_column,
                    "Churn Probability",
                    "Churn Prediction",
                    "Risk Level",
                    "High-Value Customer",
                    "Retention Priority",
                    "Risk Factors",
                    "Recommended Action"
                ]

                display_columns = [
                    column for column in display_columns
                    if column is not None and column in result_df.columns
                ]

                unified_result_df = result_df[display_columns].copy()

                # Make the high-value field easier to read.
                if "High-Value Customer" in unified_result_df.columns:
                    unified_result_df["High-Value Customer"] = (
                        unified_result_df["High-Value Customer"]
                        .map({True: "Yes", False: "No"})
                    )

                st.dataframe(
                    unified_result_df,
                    use_container_width=True,
                    hide_index=True
                )

                # =================================================
                # DOWNLOAD RESULTS
                # =================================================
                # Keep the original download options. Share as Link is an
                # additional action displayed directly underneath them.
                st.markdown("### 📄 Download Results")

                csv_bytes = result_df.to_csv(index=False).encode("utf-8")
                churners_csv_bytes = churners_df.to_csv(index=False).encode("utf-8")
                excel_bytes = create_excel(result_df, churners_df)
                pdf_bytes = create_pdf(
                    churners_df,
                    total_customers,
                    churn_count,
                    churn_rate
                )

                d1, d2, d3, d4 = st.columns(4)

                with d1:
                    st.download_button(
                        "⬇️ Full Results CSV",
                        data=csv_bytes,
                        file_name="customer_churn_predictions.csv",
                        mime="text/csv",
                        use_container_width=True,
                        key="download_full_csv"
                    )

                with d2:
                    st.download_button(
                        "⬇️ Churners CSV",
                        data=churners_csv_bytes,
                        file_name="customers_likely_to_churn.csv",
                        mime="text/csv",
                        use_container_width=True,
                        key="download_churners_csv"
                    )

                with d3:
                    st.download_button(
                        "⬇️ Excel Report",
                        data=excel_bytes,
                        file_name="customer_churn_report.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True,
                        key="download_excel"
                    )

                with d4:
                    st.download_button(
                        "⬇️ PDF Report",
                        data=pdf_bytes,
                        file_name="customer_churn_report.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                        key="download_pdf"
                    )

                # =================================================
                # SHARE AS LINK — BELOW DOWNLOADS
                # =================================================
                batch_share_payload = {
                    "result": result_df.to_dict(orient="records")
                }

                # Store the complete result server-side before rendering the
                # button, allowing the share URL to remain short.
                batch_share_id = create_share_id(batch_share_payload)

                render_copy_link_button(
                    "batch",
                    batch_share_payload,
                    "batch-share",
                    share_id=batch_share_id,
                    button_label="🔗 Share as Link"
                )

        except Exception as e:
            st.error("❌ Unable to process the uploaded file.")
            st.error(f"Error: {e}")
            # ==========================================
# 🤖 AI CUSTOMER CHURN CHATBOT
# ==========================================

st.markdown("---")

st.header("🤖 AI Customer Churn Assistant")

st.write(
    "Ask me anything about customer churn, churn prediction, "
    "risk levels, risk factors, or customer retention."
)

# -------------------------------------------------
# CHAT HISTORY
# -------------------------------------------------

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

# Display previous messages
for message in st.session_state.chat_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# -------------------------------------------------
# USER QUESTION
# -------------------------------------------------

user_question = st.chat_input(
    "Ask your question here..."
)

# -------------------------------------------------
# PROCESS QUESTION
# -------------------------------------------------

if user_question:

    # Display user's question
    with st.chat_message("user"):
        st.markdown(user_question)

    # Save user message
    st.session_state.chat_messages.append({
        "role": "user",
        "content": user_question
    })

    try:
        # -------------------------------------------------
        # CHECK OPENAI SECRET
        # -------------------------------------------------

        if "OPENAI_API_KEY" not in st.secrets:
            raise RuntimeError(
                "OPENAI_API_KEY was not found in Streamlit Secrets."
            )

        api_key = str(st.secrets["OPENAI_API_KEY"]).strip()

        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY exists, but the value is empty."
            )

        # -------------------------------------------------
        # CREATE OPENAI CLIENT
        # -------------------------------------------------

        client = OpenAI(
            api_key=api_key
        )

        # -------------------------------------------------
        # CUSTOMER CHURN ASSISTANT INSTRUCTIONS
        # -------------------------------------------------

        instructions = """
You are the AI Customer Churn Assistant inside a Customer Churn
Prediction application.

Your main topics are:
- customer churn
- churn prediction
- churn probability
- customer risk levels
- churn risk factors
- customer retention
- retention strategies
- machine learning for churn prediction

Answer clearly and simply, suitable for college students.

When explaining risk:
- LOW risk means the customer has relatively low predicted churn risk.
- MEDIUM risk means the customer needs attention.
- HIGH risk means the customer should be considered a higher priority
  for retention.

Give practical and general retention suggestions.

Do not invent customer-specific prediction results.
If the user wants a prediction for a particular customer, explain that
the prediction should be obtained from the prediction section of this
application using the customer's actual input data.

If a question is unrelated to customer churn, politely explain that
you mainly assist with the customer churn application.
"""

        # -------------------------------------------------
        # SEND QUESTION TO OPENAI
        # -------------------------------------------------

        response = client.responses.create(
            model="gpt-5.6-luna",
            instructions=instructions,
            input=user_question
        )

        # -------------------------------------------------
        # GET ANSWER
        # -------------------------------------------------

        answer = response.output_text

        if not answer or not answer.strip():
            answer = (
                "I could not generate an answer. "
                "Please try asking your question again."
            )

    except Exception as e:

        # -------------------------------------------------
        # SHOW A USEFUL ERROR
        # -------------------------------------------------

        error_text = str(e)

        if "OPENAI_API_KEY was not found" in error_text:
            answer = (
                "⚠️ OpenAI API key is missing.\n\n"
                "Go to **Manage app → Settings → Secrets** and add:\n\n"
                "```toml\n"
                'OPENAI_API_KEY = "your_api_key_here"\n'
                "```"
            )

        elif "OPENAI_API_KEY exists, but the value is empty" in error_text:
            answer = (
                "⚠️ Your `OPENAI_API_KEY` secret is empty. "
                "Please add the actual API key in Streamlit Secrets."
            )

        elif "401" in error_text or "invalid_api_key" in error_text.lower():
            answer = (
                "⚠️ The OpenAI API key was rejected. "
                "Please check that the key saved in Streamlit Secrets "
                "is correct."
            )

        elif "429" in error_text:
            answer = (
                "⚠️ The OpenAI API request was limited. "
                "Please check your API account usage and try again later."
            )

        elif "model" in error_text.lower():
            answer = (
                "⚠️ There is a problem with the OpenAI model configuration.\n\n"
                f"Technical error: `{error_text}`"
            )

        else:
            answer = (
                "⚠️ AI chatbot could not connect.\n\n"
                f"Technical error: `{error_text}`"
            )

    # -------------------------------------------------
    # DISPLAY AI ANSWER
    # -------------------------------------------------

    with st.chat_message("assistant"):
        st.markdown(answer)

    # Save AI response
    st.session_state.chat_messages.append({
        "role": "assistant",
        "content": answer
    })
