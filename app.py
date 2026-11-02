from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "loan_model.joblib"

NUMERIC_FEATURES = [
    "no_of_dependents",
    "income_annum",
    "loan_amount",
    "loan_term",
    "cibil_score",
    "residential_assets_value",
    "commercial_assets_value",
    "luxury_assets_value",
    "bank_asset_value",
]
CATEGORICAL_FEATURES = ["education", "self_employed"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="🏦",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp { background: #f3f6f7; }
    .block-container { max-width: 1100px; padding-top: 2.2rem; }
    div[data-testid="stForm"] {
        background: white; padding: 1.3rem; border: 1px solid #dce5e8;
        border-radius: 14px;
    }
    .hero {
        background: #ffffff; padding: 1.5rem 1.7rem; border-radius: 16px;
        border: 1px solid #dce5e8; margin-bottom: 1.2rem;
    }
    .muted { color: #5b6b73; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🏦 Loan Approval Prediction")
st.markdown(
    '<div class="hero"><h3>Check a loan application</h3>'
    '<p class="muted">Enter applicant, loan and asset details. '
    'The trained machine-learning model will estimate the likelihood of approval.</p>'
    '</div>',
    unsafe_allow_html=True,
)

@st.cache_resource
def load_model_from_path(path_string: str):
    return joblib.load(path_string)

def get_model():
    if MODEL_PATH.is_file():
        try:
            return load_model_from_path(str(MODEL_PATH)), None
        except Exception as exc:
            return None, (
                "The model file exists, but it could not be loaded. "
                "Check that it was created by the supplied Colab notebook and "
                "that scikit-learn versions are compatible. "
                f"Technical detail: {type(exc).__name__}: {exc}"
            )

    st.warning("The trained model is not included in this app yet.")
    uploaded_model = st.file_uploader(
        "Upload your trained loan_model.joblib file",
        type=["joblib", "pkl"],
        help="Train the model in Google Colab, download it, then upload it here.",
    )
    if uploaded_model is not None:
        try:
            return joblib.load(uploaded_model), None
        except Exception as exc:
            return None, f"Could not load the uploaded model: {type(exc).__name__}: {exc}"
    return None, "Upload the trained model file above to enable predictions."

with st.sidebar:
    st.header("About this project")
    st.write(
        "This student project uses a Random Forest machine-learning pipeline "
        "trained on a historical loan-approval dataset."
    )
    st.caption(
        "Predictions are educational estimates only. They are not financial "
        "advice or an actual bank decision."
    )

left, right = st.columns([1.35, 0.85], gap="large")

with left:
    with st.form("loan_details"):
        st.subheader("1. Applicant details")
        c1, c2 = st.columns(2)
        with c1:
            no_of_dependents = st.number_input(
                "Number of dependents", min_value=0, max_value=20, value=2, step=1
            )
            education = st.selectbox("Education", ["Graduate", "Not Graduate"])
            income_annum = st.number_input(
                "Annual income (₹)", min_value=0, value=6000000, step=100000
            )
        with c2:
            self_employed = st.selectbox("Self-employed", ["No", "Yes"])
            cibil_score = st.number_input(
                "CIBIL score", min_value=300, max_value=900, value=750, step=1
            )
            loan_term = st.number_input(
                "Loan term (years)", min_value=1, max_value=40, value=12, step=1
            )

        st.divider()
        st.subheader("2. Loan details")
        loan_amount = st.number_input(
            "Loan amount (₹)", min_value=0, value=15000000, step=100000
        )

        st.divider()
        st.subheader("3. Asset details (₹)")
        a1, a2 = st.columns(2)
        with a1:
            residential_assets_value = st.number_input(
                "Residential assets", min_value=0, value=5000000, step=100000
            )
            luxury_assets_value = st.number_input(
                "Luxury assets", min_value=0, value=10000000, step=100000
            )
        with a2:
            commercial_assets_value = st.number_input(
                "Commercial assets", min_value=0, value=3000000, step=100000
            )
            bank_asset_value = st.number_input(
                "Bank assets", min_value=0, value=4000000, step=100000
            )

        submitted = st.form_submit_button(
            "Predict loan approval", type="primary", use_container_width=True
        )

with right:
    st.subheader("Prediction result")
    st.info("Complete the form and select **Predict loan approval**.")

    if submitted:
        row = {
            "no_of_dependents": float(no_of_dependents),
            "income_annum": float(income_annum),
            "loan_amount": float(loan_amount),
            "loan_term": float(loan_term),
            "cibil_score": float(cibil_score),
            "residential_assets_value": float(residential_assets_value),
            "commercial_assets_value": float(commercial_assets_value),
            "luxury_assets_value": float(luxury_assets_value),
            "bank_asset_value": float(bank_asset_value),
            "education": education,
            "self_employed": self_employed,
        }
        input_df = pd.DataFrame([row], columns=FEATURES)

        model, model_error = get_model()
        if model is None:
            st.error(model_error)
        else:
            try:
                probabilities = model.predict_proba(input_df)[0]
                classes = list(getattr(model, "classes_", []))
                approved_index = classes.index(1) if 1 in classes else 1
                if approved_index >= len(probabilities):
                    raise ValueError("The model does not contain an approval class.")

                probability = float(probabilities[approved_index]) * 100
                if probability >= 50:
                    st.success("Likely approved")
                else:
                    st.error("Likely rejected")
                st.metric("Estimated approval probability", f"{probability:.1f}%")
                st.progress(min(max(int(round(probability)), 0), 100))
                st.caption(
                    "This is a prediction from historical training data, "
                    "not a guarantee or an actual lender decision."
                )
            except Exception as exc:
                st.error(
                    "Prediction could not be completed. "
                    f"Technical detail: {type(exc).__name__}: {exc}"
                )

st.divider()
st.caption("Educational project • Python • Streamlit • scikit-learn")
