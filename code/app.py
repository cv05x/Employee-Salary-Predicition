import streamlit as st
import joblib 
import numpy as np
import pandas as pd

model1 = joblib.load("XGBoost_Algorithm.pkl")
encoders = joblib.load("label_encoders.pkl")

st.title("Employee Salary Prediction")
st.divider()

st.write("Using this app, predict whether an employee earns above 50K or not")

age = st.number_input("Age", min_value=17, max_value=75)
workclass = st.selectbox("Workclass", encoders['workclass'].classes_)
educational_num = st.slider("Educational Num", 1, 16)
marital_status = st.selectbox("Marital Status", encoders['marital-status'].classes_)
occupation = st.selectbox("Occupation", encoders['occupation'].classes_)
relationship = st.selectbox("Relationship", encoders['relationship'].classes_)  
race = st.selectbox("Race", encoders['race'].classes_)
gender = st.selectbox("Gender", encoders['gender'].classes_)
capital_gain = st.number_input("Capital Gain", value=0, min_value=0)
capital_loss = st.number_input("Capital Loss", value=0, min_value=0)
hours_per_week = st.number_input("Hours Per Week", min_value=1, max_value=100)
native_country = st.selectbox("Native Country", encoders['native-country'].classes_)

input_data = {
    'age': age,
    'workclass': encoders['workclass'].transform([workclass])[0],
    'educational-num': educational_num,
    'marital-status': encoders['marital-status'].transform([marital_status])[0],
    'occupation': encoders['occupation'].transform([occupation])[0],
    'relationship': encoders['relationship'].transform([relationship])[0],
    'race': encoders['race'].transform([race])[0],
    'gender': encoders['gender'].transform([gender])[0],
    'capital-gain': capital_gain,
    'capital-loss': capital_loss,
    'hours-per-week': hours_per_week,
    'native-country': encoders['native-country'].transform([native_country])[0]
}

input_df = pd.DataFrame([input_data])

st.divider()

if st.button("Predict Income Category"):
    prediction = model1.predict(input_df)[0]
    result = ">50K" if prediction == 1 else "<=50K"
    st.success(f"Predicted Income: {result}")
