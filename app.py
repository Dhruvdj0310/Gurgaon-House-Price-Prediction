import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ============================================================
# STREAMLIT UI ONLY
# Your original ML training/inference logic is NOT changed.
# This app uses the already generated model.pkl and pipeline.pkl
# ============================================================

MODEL_FILE = "model.pkl"
PIPELINE_FILE = "pipeline.pkl"

st.set_page_config(
    page_title="Gurgaon House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------- Custom CSS --------------------
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #111827 55%, #172554 100%);
        color: #f8fafc;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #111827, #0f172a);
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    .hero {
        padding: 28px 32px;
        border-radius: 24px;
        background: linear-gradient(135deg, rgba(37,99,235,.25), rgba(124,58,237,.20));
        border: 1px solid rgba(255,255,255,.12);
        box-shadow: 0 15px 45px rgba(0,0,0,.25);
        margin-bottom: 25px;
    }

    .hero h1 {
        font-size: 42px;
        margin-bottom: 5px;
        color: white;
    }

    .hero p {
        font-size: 17px;
        color: #cbd5e1;
        margin-top: 0;
    }

    .section-card {
        padding: 22px;
        border-radius: 20px;
        background: rgba(255,255,255,.055);
        border: 1px solid rgba(255,255,255,.10);
        margin-bottom: 20px;
    }

    .section-title {
        font-size: 21px;
        font-weight: 700;
        color: #e2e8f0;
        margin-bottom: 14px;
    }

    .prediction-card {
        padding: 28px;
        border-radius: 22px;
        background: linear-gradient(135deg, rgba(16,185,129,.18), rgba(14,165,233,.14));
        border: 1px solid rgba(52,211,153,.28);
        text-align: center;
        margin-top: 15px;
    }

    .prediction-label {
        color: #94a3b8;
        font-size: 15px;
        margin-bottom: 5px;
    }

    .prediction-value {
        font-size: 42px;
        font-weight: 800;
        color: #6ee7b7;
    }

    .info-box {
        padding: 14px 17px;
        border-radius: 14px;
        background: rgba(59,130,246,.10);
        border: 1px solid rgba(96,165,250,.18);
        color: #cbd5e1;
        font-size: 14px;
    }

    .footer {
        text-align: center;
        color: #64748b;
        padding: 25px 0 10px 0;
        font-size: 13px;
    }

    div.stButton > button {
        width: 100%;
        border-radius: 12px;
        height: 48px;
        font-weight: 700;
        border: none;
    }
</style>
""", unsafe_allow_html=True)


# -------------------- Load Model --------------------
if not os.path.exists(MODEL_FILE) or not os.path.exists(PIPELINE_FILE):
    st.error(
        "⚠️ model.pkl or pipeline.pkl not found. "
        "Run your original Python model file first so that these files are created."
    )
    st.stop()

try:
    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)
except Exception as e:
    st.error(f"Could not load the saved model/pipeline: {e}")
    st.stop()


# -------------------- Header --------------------
st.markdown("""
<div class="hero">
    <h1>🏠 House Price Predictor</h1>
    <p>Machine Learning powered house-value prediction using your Random Forest model.</p>
</div>
""", unsafe_allow_html=True)


# -------------------- Sidebar --------------------
with st.sidebar:
    st.markdown("## 🏠 Property Details")
    st.caption("Enter the property information below.")

    st.divider()

    longitude = st.number_input(
        "📍 Longitude",
        value=-122.23,
        min_value=-180.0,
        max_value=180.0,
        step=0.01
    )

    latitude = st.number_input(
        "📍 Latitude",
        value=37.88,
        min_value=-90.0,
        max_value=90.0,
        step=0.01
    )

    housing_median_age = st.number_input(
        "🏗️ Housing Median Age",
        value=30.0,
        min_value=0.0,
        step=1.0
    )

    total_rooms = st.number_input(
        "🚪 Total Rooms",
        value=2000.0,
        min_value=0.0,
        step=100.0
    )

    total_bedrooms = st.number_input(
        "🛏️ Total Bedrooms",
        value=400.0,
        min_value=0.0,
        step=10.0
    )

    population = st.number_input(
        "👨‍👩‍👧 Population",
        value=1000.0,
        min_value=0.0,
        step=50.0
    )

    households = st.number_input(
        "🏘️ Households",
        value=350.0,
        min_value=0.0,
        step=10.0
    )

    median_income = st.number_input(
        "💰 Median Income",
        value=4.0,
        min_value=0.0,
        step=0.1
    )

    ocean_proximity = st.selectbox(
        "🌊 Ocean Proximity",
        ["<1H OCEAN", "INLAND", "NEAR OCEAN", "NEAR BAY", "ISLAND"]
    )

    predict_button = st.button("🔮 Predict House Price", type="primary")


# -------------------- Main Content --------------------
left, right = st.columns([1.35, 1])

with left:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-title">📊 Property Summary</div>',
        unsafe_allow_html=True
    )

    summary_col1, summary_col2, summary_col3 = st.columns(3)

    summary_col1.metric("Rooms", f"{total_rooms:,.0f}")
    summary_col2.metric("Bedrooms", f"{total_bedrooms:,.0f}")
    summary_col3.metric("Households", f"{households:,.0f}")

    summary_col1, summary_col2, summary_col3 = st.columns(3)

    summary_col1.metric("Population", f"{population:,.0f}")
    summary_col2.metric("Median Age", f"{housing_median_age:.0f} yrs")
    summary_col3.metric("Income", f"{median_income:.2f}")

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        💡 <b>How it works:</b> Your inputs are converted into a DataFrame,
        passed through the saved preprocessing pipeline, and then sent to
        your trained Random Forest model.
    </div>
    """, unsafe_allow_html=True)


with right:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-title">🤖 Model Status</div>',
        unsafe_allow_html=True
    )

    st.success("Model loaded successfully")
    st.write("**Algorithm:** Random Forest Regressor")
    st.write("**Preprocessing:** Saved Pipeline")
    st.write("**Categorical Feature:** Ocean Proximity")
    st.write("**Prediction Type:** House Value")

    st.markdown("</div>", unsafe_allow_html=True)


# -------------------- Prediction --------------------
if predict_button:

    # IMPORTANT:
    # These column names match the columns expected by your
    # original preprocessing pipeline.
    input_data = pd.DataFrame([{
        "longitude": longitude,
        "latitude": latitude,
        "housing_median_age": housing_median_age,
        "total_rooms": total_rooms,
        "total_bedrooms": total_bedrooms,
        "population": population,
        "households": households,
        "median_income": median_income,
        "ocean_proximity": ocean_proximity
    }])

    try:
        transformed_input = pipeline.transform(input_data)
        prediction = model.predict(transformed_input)[0]

        st.markdown(f"""
        <div class="prediction-card">
            <div class="prediction-label">Estimated House Value</div>
            <div class="prediction-value">${prediction:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)

        st.balloons()

        with st.expander("🔍 View Input Data"):
            st.dataframe(input_data, use_container_width=True)

    except Exception as e:
        st.error(f"Prediction failed: {e}")


# -------------------- Footer --------------------
st.markdown("""
<div class="footer">
    Built with ❤️ using Python • Scikit-learn • Joblib • Streamlit
</div>
""", unsafe_allow_html=True)
