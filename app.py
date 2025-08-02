import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load('heart_model.pkl')
scaler = joblib.load('heart_scaler.pkl')

st.title("❤️ Heart Disease Prediction App")

st.write("Please enter the patient's information:")

age = st.number_input("Age", min_value=20, max_value=100, value=50)
sex = st.selectbox("Sex", options=[("Male", 1), ("Female", 0)])
cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure", value=120)
chol = st.number_input("Cholesterol", value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120", [0, 1])
restecg = st.selectbox("Resting ECG", [0, 1, 2])
thalach = st.number_input("Max Heart Rate", value=150)
exang = st.selectbox("Exercise Induced Angina", [0, 1])
oldpeak = st.number_input("Oldpeak", value=1.0)
slope = st.selectbox("Slope", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels", [0, 1, 2, 3])
thal = st.selectbox("Thal", [0, 1, 2, 3])

# Prepare data
data = np.array([[age, sex[1], cp, trestbps, chol, fbs, restecg,
                  thalach, exang, oldpeak, slope, ca, thal]])
scaled = scaler.transform(data)

# Predict
if st.button("🔍 Predict"):
    result = model.predict(scaled)[0]
    if result == 1:
        st.error("⚠️ Likely Heart Disease Detected")
    else:
        st.success("✅ Likely Healthy")
