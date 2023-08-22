#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 10:04:34 2023

@author: ellietripathi
"""

import numpy as np
import pickle
import streamlit as st


# loading the saved model.
loaded_model = pickle.load(open('/Users/ellietripathi/Downloads/trained_model.sav', 'rb'))

# Creating a function for prediction.
def heart_disease_prediction(input_data):

    # change the input data to numpy array.
    input_data = np.asarray(input_data)

    # making prediction for this particular input.
    input_data_reshaped = input_data.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if prediction[0] == 0:
        return "You dont have a Heart Disease."
    else:
        return "You have a Heart Disease."
 
    
    
def main():
    # title for the webpage.
    st.title('Heart Disease Predictor')
    
    # getting input data from the user.   
    age = st.number_input('Age')
    sex = st.number_input('Sex')
    cp = st.number_input('Chest Pain Type')
    trestbps = st.number_input('Resting Blood Pressure')
    chol = st.number_input('Serum Cholestrol')
    fbs = st.number_input('Fasting Blood Sugar')
    restecg = st.number_input('CResting Electrocardiographic results')
    thalach = st.number_input('Maximum Heart Rate Achieved')
    exang = st.number_input('Exercise induced Anginea')
    oldpeak = st.number_input('ST depression induced by exercise')
    slope = st.number_input('Slope of the peak exercise segment')
    ca = st.number_input('Number of major vessels')
    thal = st.number_input('Type of defect')
    
    # code for prediction.
    diagnosis = ''
    
    # creating a button for prediction.
    if st.button('Test Result'):
        diagnosis = heart_disease_prediction([age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal])
        
    st.success(diagnosis)
    
    
    
    
if __name__ == '__main__':
    main()
    
    
    