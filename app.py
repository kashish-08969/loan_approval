from pathlib import Path
from io import BytesIO
import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "loan_model.joblib"

NUMERIC = [
    "no_of_dependents", "income_annum", "loan_amount", "loan_term", "cibil_score",
    "residential_assets_value", "commercial_assets_value",
    "luxury_assets_value", "bank_asset_value",
]
CATEGORICAL = ["education", "self_employed"]
FEATURES = NUMERIC + CATEGORICAL

st.set_page_config(page_title="Loan Approval Predictor", page_icon="🏦", layout="wide")
st.title("🏦 Loan Approval Prediction")
st.write("Enter applicant, loan and asset details to estimate the likelihood of loan approval.")

@st.cache_resource
def load_uploaded_model(model_bytes):
    # Only load model files from a trusted source.
    return joblib.load(BytesIO(model_bytes))

@st.cache_resource
def load_repository_model(path, modified_time):
    return joblib.load(path)

def get_model(uploaded):
    if uploaded is not None:
        return load_uploaded_model(uploaded.getvalue())
    if MODEL_PATH.is_file():
        return load_repository_model(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)
    return None

left, right = st.columns([1.35, 0.85], gap="large")

with right:
    st.subheader("Prediction result")
    uploaded_model = st.file_uploader(
        "Upload your trained loan_model.joblib or loan_model.pkl",
        type=["joblib", "pkl"],
        key="model_upload",
    )
    if uploaded_model is not None:
        st.success(f"Selected: {uploaded_model.name}")
    elif MODEL_PATH.is_file():
        st.success("Model file found in the project.")
    else:
        st.warning("Upload the trained model before requesting a prediction.")

with left:
    with st.form("loan_details"):
        st.subheader("1. Applicant details")
        a, b = st.columns(2)
        with a:
            no_of_dependents = st.number_input("Number of dependents", 0, 20, 2)
            education = st.selectbox("Education", ["Graduate", "Not Graduate"])
            income_annum = st.number_input("Annual income (₹)", min_value=0, value=6000000, step=100000)
        with b:
            self_employed = st.selectbox("Self-employed", ["No", "Yes"])
            cibil_score = st.number_input("CIBIL score", 300, 900, 750)
            loan_term = st.number_input("Loan term (years)", 1, 40, 12)

        st.divider()
        st.subheader("2. Loan details")
        loan_amount = st.number_input("Loan amount (₹)", min_value=0, value=15000000, step=100000)

        st.divider()
        st.subheader("3. Asset details (₹)")
        c, d = st.columns(2)
        with c:
            residential_assets_value = st.number_input("Residential assets", min_value=0, value=5000000, step=100000)
            luxury_assets_value = st.number_input("Luxury assets", min_value=0, value=10000000, step=100000)
        with d:
            commercial_assets_value = st.number_input("Commercial assets", min_value=0, value=3000000, step=100000)
            bank_asset_value = st.number_input("Bank assets", min_value=0, value=4000000, step=100000)

        submitted = st.form_submit_button("Predict loan approval", type="primary", use_container_width=True)

with right:
    if not submitted:
        st.info("Upload your model, complete the form, then click Predict loan approval.")
    else:
        try:
            model = get_model(uploaded_model)
            if model is None:
                st.error("No trained model found. Upload the model above.")
            else:
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
                df = pd.DataFrame([row], columns=FEATURES)
                probs = model.predict_proba(df)[0]
                classes = list(getattr(model, "classes_", []))
                approved_index = classes.index(1) if 1 in classes else 1
                probability = float(probs[approved_index]) * 100

                if probability >= 50:
                    st.success("Likely approved")
                else:
                    st.error("Likely rejected")
                st.metric("Estimated approval probability", f"{probability:.1f}%")
                st.progress(min(100, max(0, int(round(probability)))))
                st.caption("Educational estimate only; not an actual bank decision.")
        except Exception as exc:
            st.error(f"Prediction failed: {type(exc).__name__}: {exc}")
            st.info("If this is a model-version error, match the scikit-learn version used in Colab with the version in Streamlit Cloud.")

st.divider()
st.caption("Educational project • Python • Streamlit • scikit-learn")
