import streamlit as st
import joblib
import pandas as pd

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
# PREDICTION MODE
# =========================================================

# IMPORTANT:
# The prediction mode is stored in the URL so the browser's
# Back / Forward buttons can navigate between the home screen
# and the selected prediction screen on laptop/desktop.
#
# Mobile users also get an in-app Back button.

url_mode = st.query_params.get("mode")

if url_mode in ("single", "batch"):
    st.session_state.prediction_mode = url_mode
else:
    st.session_state.prediction_mode = None


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

            st.query_params["mode"] = "single"
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

            st.query_params["mode"] = "batch"
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
        st.query_params.clear()
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


# =========================================================
# BATCH PREDICTION MODE
# =========================================================

elif st.session_state.prediction_mode == "batch":

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
        st.query_params.clear()
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
        "Upload customer data",
        type=["csv", "xlsx", "json"],
        help="Supported input formats: CSV, XLSX, JSON"
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
                        mime=(
                            "application/vnd.openxmlformats-officedocument."
                            "spreadsheetml.sheet"
                        ),
                        use_container_width=True,
                        key="download_excel_report"
                    )

                with d4:
                    st.download_button(
                        "⬇️ PDF Report",
                        data=pdf_bytes,
                        file_name="customer_churn_report.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                        key="download_pdf_report"
                    )

        except Exception as e:
            st.error("❌ Unable to process the uploaded file.")
            st.error(f"Error: {e}")

