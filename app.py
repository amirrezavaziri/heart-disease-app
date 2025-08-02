import streamlit as st
import numpy as np
import joblib

# Import these if the model was trained with XGBoost or LightGBM
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Load model and scaler
model = joblib.load('heart_model.pkl')
scaler = joblib.load('heart_scaler.pkl')

st.title("❤️ Heart Disease Prediction App")

st.write("Please enter the patient's medical information:")

# User inputs
age = st.number_input("Age", min_value=20, max_value=100, value=50)
sex = st.selectbox("Sex", options=[("Male", 1), ("Female", 0)])
cp = st.selectbox("Chest Pain Type (0: Typical Angina, 1: Atypical, 2: Non-anginal, 3: Asymptomatic)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure (mm Hg)", value=120)
chol = st.number_input("Serum Cholesterol (mg/dl)", value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1 = True; 0 = False)", [0, 1])
restecg = st.selectbox("Resting ECG Results (0, 1, 2)", [0, 1, 2])
thalach = st.number_input("Maximum Heart Rate Achieved", value=150)
exang = st.selectbox("Exercise Induced Angina (1 = Yes; 0 = No)", [0, 1])
oldpeak = st.number_input("ST Depression Induced by Exercise", value=1.0)
slope = st.selectbox("Slope of the Peak Exercise ST Segment (0, 1, 2)", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels (0–3) Colored by Fluoroscopy", [0, 1, 2, 3])
thal = st.selectbox("Thalassemia (0 = Normal; 1 = Fixed Defect; 2 = Reversible Defect; 3 = Unknown)", [0, 1, 2, 3])

# Prepare input for prediction
data = np.array([[age, sex[1], cp, trestbps, chol, fbs, restecg,
                  thalach, exang, oldpeak, slope, ca, thal]])
scaled = scaler.transform(data)

# Make prediction
if st.button("🔍 Predict"):
    result = model.predict(scaled)[0]
    if result == 1:
        st.error("⚠️ The model predicts a high risk of heart disease.")
    else:
        st.success("✅ The model predicts a low risk of heart disease.")

