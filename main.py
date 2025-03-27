import streamlit as st
import numpy as np
import pandas as pd
import joblib

insurance1 = joblib.load("insurance.joblib")

st.title('Estimated Insurance Price')

age = st.slider('Enter age', 0,100)
sex = st.selectbox('Gender', ['Male', 'Female'])
bmi = st.slider('bmi', 0,50)
children = st.slider('childrem', 0,5)
smoker = st.selectbox('smoker', ['yes', 'no'])
region = st.selectbox('region', ['northeast', 'northwest', 'southeast', 'southwest'])

input={'age' : age, 'sex': sex, 'bmi' : bmi, 'children' : children, 'smoker' : smoker, 'region' : region}


cat_cols = ['sex', 'smoker', 'region']
numeric_cols= ['age', 'bmi', 'children']
encoded_cols = ['sex_female',
 'sex_male',
 'smoker_no',
 'smoker_yes',
 'region_northeast',
 'region_northwest',
 'region_southeast',
 'region_southwest']

def predict():
    input_df = pd.DataFrame([input])
    input_df[numeric_cols] = insurance1['scaler'].transform(input_df[numeric_cols])
    input_df[encoded_cols] = insurance1['encoder'].transform(input_df[cat_cols])
    X_input = input_df[numeric_cols + encoded_cols]
    pred = insurance1['model'].predict(X_input)
    return pred

st.button('Estimate', on_click=predict)

st.write('Prediction', predict())












