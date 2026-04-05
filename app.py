import numpy as np
import pandas as pd
import streamlit as st
import joblib

scaler=joblib.load("scale.pkl")
le_gender=joblib.load("Label Encodinggender.pkl")
model=joblib.load("Random_Forest_model.pkl")
sm=joblib.load("Label Encodingsmoker.pkl")
dia=joblib.load("Label Encodingdiabetic.pkl")


st.set_page_config(page_title=" Insurance Claim Predictor",layout="centered")
st.title("Health Insurance Predictin App")
st.write("Enter the details below")


with st.form("input form:"):
    col1,col2=st.columns(2)
    with col1:
        age=st.number_input("Age",min_value=0,max_value=100,value=30)
        bmi=st.number_input("BMI",min_value=10.0,max_value=60.0,value=20.0)
        children=st.number_input("Children",min_value=0,max_value=8,value=0)
    with col2:
        blood_pressure=st.number_input("BloodPressure",min_value=60.0,max_value=120.0)
        gender=st.selectbox("Gender",options=le_gender.classes_)
        diabetic=st.selectbox("Diabetic",options=dia.classes_)
        smoker=st.selectbox("Smoker",options=sm.classes_)
        
    submitted=st.form_submit_button("Predict Payment")



if submitted:

    input_data=pd.DataFrame({

    "age":[age],
    "gender":[gender],
    "bmi":[bmi],
    "bloodpressure":[blood_pressure],
    "diabetic":[diabetic],
    "smoker":[smoker],
    "children":[children],
    
    })
# encoding
    input_data["gender"] = le_gender.transform(input_data["gender"])
    input_data["diabetic"] = dia.transform(input_data["diabetic"])
    input_data["smoker"] = sm.transform(input_data["smoker"])

    # scaling (only numeric columns used in training scaler)
    num_cols = ['age','bmi','bloodpressure']
    input_data[num_cols] = scaler.transform(input_data[num_cols])

    # EXACT training order (VERY IMPORTANT)
    input_data = input_data[['age','gender','bmi','bloodpressure','diabetic','smoker','children']]

    # prediction
    prediction = model.predict(input_data)[0]

    st.success(f"**Estimated Insurance Payment:** {prediction:,.2f}")

    