import streamlit as st
import joblib
import pandas as pd

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(
    page_title="Airline Recommendation Predictor",
    page_icon="✈️",
    layout="wide"
)

# ----------------------------
# CUSTOM CSS
# ----------------------------
st.markdown("""
<style>

body {
    background: #e7f1ff;
}

.main-container {
    background: rgba(255, 255, 255, 0.6);
    padding: 25px;
    border-radius: 15px;
    backdrop-filter: blur(10px);
}

h1 {
    color: #0a4fa3;
    text-align: center;
    font-weight: 800;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #2b2b2b;
    margin-top: -10px;
    margin-bottom: 25px;
}

.feature-box {
    background: white;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #d9d9d9;
    box-shadow: 0 2px 6px rgba(0,0,0,0.08);
}

.feature-title {
    font-weight: 700;
    color: #0a4fa3;
    margin-bottom: 10px;
    text-align:center;
}

.predict-btn button {
    background-color: #0a4fa3 !important;
    color: white !important;
    font-size: 18px;
    border-radius: 12px;
    padding: 10px;
}

.card {
    background: white;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #e0e0e0;
    box-shadow: 0 2px 6px rgba(0,0,0,0.07);
}

.footer {
    text-align:center;
    padding: 15px;
    margin-top: 30px;
    color: gray;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)


# ----------------------------
# LOAD MODEL
# ----------------------------
try:
    model = joblib.load("best_xgb_model.joblib")
except FileNotFoundError:
    st.error("❌ Model file not found. Place 'best_xgb_model.joblib' next to app.py.")
    st.stop()


# ----------------------------
# HEADER
# ----------------------------
st.markdown("<h1>✈️ Airline Recommendation Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Predict whether the customer will recommend the airline.</p>", unsafe_allow_html=True)

st.markdown("<div class='main-container'>", unsafe_allow_html=True)

# ----------------------------
# FEATURE SELECTION
# ----------------------------

st.markdown("<h3 class='feature-title'>🎚 Feature Ratings</h3>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("<div class='feature-box'>", unsafe_allow_html=True)
    seat_comfort = st.slider("🪑 Seat Comfort", 1, 5, 3)
    cabin_service = st.slider("👨‍✈️ Cabin Service", 1, 5, 3)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='feature-box'>", unsafe_allow_html=True)
    food_bev = st.slider("🍽 Food & Beverage", 1, 5, 3)
    entertainment = st.slider("🎬 Entertainment", 1, 5, 3)
    st.markdown("</div>", unsafe_allow_html=True)

with col3:
    st.markdown("<div class='feature-box'>", unsafe_allow_html=True)
    ground_service = st.slider("🛄 Ground Service", 1, 5, 3)
    st.markdown("</div>", unsafe_allow_html=True)


# Prepare model input
input_data = pd.DataFrame(
    [[seat_comfort, food_bev, cabin_service, entertainment, ground_service]],
    columns=['seat_comfort', 'food_bev', 'cabin_service', 'entertainment', 'ground_service']
)

# ----------------------------
# PREDICT BUTTON
# ----------------------------
st.markdown("<div class='predict-btn'>", unsafe_allow_html=True)
predict = st.button("🔍 Predict Recommendation", use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

st.write("")

# ----------------------------
# SHOW RESULT
# ----------------------------
if predict:
    with st.spinner("Analyzing..."):
        pred = model.predict(input_data)
        proba = model.predict_proba(input_data)[0]

    result_col1, result_col2 = st.columns([1, 1.2])

    with result_col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("📊 Prediction")

        if pred[0] == 1:
            st.success("### 👍 Recommended")
            st.metric("Probability", f"{proba[1]:.2f}")
        else:
            st.error("### 👎 Not Recommended")
            st.metric("Probability", f"{proba[0]:.2f}")

        st.markdown("</div>", unsafe_allow_html=True)

    with result_col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("🔧 Feature Values Used")
        st.dataframe(input_data.style.highlight_max(axis=1))
        st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)


# ----------------------------
# FOOTER
# ----------------------------
st.markdown("<p class='footer'>Built with Streamlit • XGBoost Model • Professional Blue UI</p>", unsafe_allow_html=True)
