import pickle
import os

try:
    import streamlit as st  # type: ignore[import]
except ImportError:
    raise RuntimeError("The required package 'streamlit' is not installed. Please install it with `pip install streamlit`.")
except ModuleNotFoundError:
    raise RuntimeError("The required package 'streamlit' is not installed. Please install it with `pip install streamlit`.")

try:
    import numpy as np  # type: ignore[import]
except ImportError:
    raise RuntimeError("The required package 'numpy' is not installed. Please install it with `pip install numpy`.")

from feature_extractor import extract_features

# Load models
MODEL_DIR = "models"

with open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)

with open(os.path.join(MODEL_DIR, "random_forest.pkl"), "rb") as f:
    rf = pickle.load(f)

CLASS_NAMES = ["Normal", "Inner Race Fault", "Outer Race Fault", "Ball Fault"]

st.set_page_config(page_title="Bearing Fault Detection", layout="centered")

st.title("⚙️ Bearing Fault Detection System")
st.write("Upload vibration signal or generate sample data to detect faults.")

if np is None:
    st.error("The required package 'numpy' is not installed. Please install it with `pip install numpy`.")
    st.stop()

# Option select
option = st.selectbox("Choose Input Type", ["Generate Sample Data", "Upload CSV"])

# Generate sample
if option == "Generate Sample Data":
    st.subheader("Generate Random Signal")

    if st.button("Generate & Predict"):
        # random signal (1024 samples)
        signal = np.random.randn(1024)

        features = extract_features(signal)
        fv = np.array(list(features.values())).reshape(1, -1)
        fv_scaled = scaler.transform(fv)

        pred = rf.predict(fv_scaled)[0]
        prob = rf.predict_proba(fv_scaled)[0]

        st.success(f"Prediction: {CLASS_NAMES[pred]}")
        st.write("Confidence:", round(max(prob)*100, 2), "%")

# Upload CSV
elif option == "Upload CSV":
    uploaded_file = st.file_uploader("Upload CSV with 1024 signal values", type=["csv"])

    if uploaded_file is not None:
        data = np.loadtxt(uploaded_file, delimiter=",")

        if len(data) != 1024:
            st.error("CSV must contain exactly 1024 values")
        else:
            features = extract_features(data)
            fv = np.array(list(features.values())).reshape(1, -1)
            fv_scaled = scaler.transform(fv)

            pred = rf.predict(fv_scaled)[0]
            prob = rf.predict_proba(fv_scaled)[0]

            st.success(f"Prediction: {CLASS_NAMES[pred]}")
            st.write("Confidence:", round(max(prob)*100, 2), "%")