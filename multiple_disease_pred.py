import joblib
import streamlit as st 
from streamlit_option_menu import option_menu 
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(filename='app.log', level=logging.INFO)

# Log an info message
logging.info('User accessed the app.')

# Log an error message
try:
    # Load the models
    diabetes_model = joblib.load(open('diabetes_model.joblib', 'rb'))
    heart_model = joblib.load(open('heart_model.joblib', 'rb'))
    breast_model = joblib.load(open('breast_cancer_model.joblib', 'rb'))
    parkinson_model = joblib.load(open('parkinson_model.joblib', 'rb'))
    
    with st.sidebar:
        selected = option_menu('Multiple Disease Prediction System',
                            ['Diabetes Disease Prediction',
                            'Heart Disease Prediction',
                            'Breast Cancer Prediction',
                            'Parkinson Disease Prediction'],
                            icons=['activity','heart-pulse','person'], 
                            default_index=0)

    ### Diabetes Disease -----------------------------------------------
    if selected == 'Diabetes Disease Prediction':
        st.title(':medical_symbol: Diabetes Disease Prediction') # Page title
        st.subheader('Please Read the Medical description below before entering the values :exclamation:')
        
        # User inputs
        col1, col2, col3 = st.columns(3)
        
        with col1:
            Pregnancies = st.text_input("Number of times pregnant")
        with col2:
            Glucose = st.text_input("Glucose concentration")
        with col3:
            BloodPressure = st.text_input("Blood pressure (mm Hg)")
        with col1:
            SkinThickness = st.text_input("Triceps skin fold thickness (mm)")
        with col2:
            Insulin = st.text_input("Insulin (mu U/ml)")
        with col3:
            BMI = st.text_input("Body mass index")
        with col1:
            DiabetesPedigreeFunction = st.text_input("Diabetes pedigree function value")
        with col2:
            Age = st.text_input("Age of the patient")
        
        diab_diagnosis = ''
        
        if st.button("Diabetes Test result"):
            try:
                features = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
                features = [float(feature) for feature in features]

                diabete_pred = diabetes_model.predict([features])

                if diabete_pred == 1:
                    diab_diagnosis = "The person has a high probability to be Diabetic"
                else:
                    diab_diagnosis = "The person has less chance to be Diabetic"

            except Exception as e:
                st.error(f"Error predicting diabetes: {e}")
                diab_diagnosis = "Prediction error"

        st.success(diab_diagnosis)
        
        st.header('Attention! Please follow the values given in the range') 
        st.markdown("""
            * **Pregnancies**: Number of times pregnant, range: [0-17]
            * **Glucose**: Plasma glucose concentration a 2 hours in an oral glucose tolerance test, range: [30.46-199]
            * **BloodPressure**: Diastolic blood pressure (mm Hg), range: [24-122]
            * **SkinThickness**: Triceps skin fold thickness (mm), range: [7-99]
            * **Insulin**: 2-Hour serum insulin (mu U/ml), range: [89.10-846]
            * **BMI**: Body mass index (weight in kg/(height in m)^2), range: [18, 67]
            * **DiabetesPedigreeFunction**: Diabetes pedigree function. The range value is [0-2.5]
            * **Age**: Age (years)
        """)

    ### Heart Disease -----------------------------------------
    if selected == 'Heart Disease Prediction':
        st.title(":heartpulse: Heart Disease Prediction ") # Page title
        st.subheader("Please read the Medical descriptions below before entering the values :exclamation:")
        
        # User inputs
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.text_input("Age of the person")
        with col2:
            sex = st.text_input("Sex: 0 or 1?")
        with col3:
            cp = st.text_input("Chest pain value")
        with col1:
            trestbps = st.text_input("Resting blood pressure value")
        with col2:
            chol = st.text_input("Serum cholesterol in mg/dl value")
        with col3:
            fbs = st.text_input("Fasting blood sugar")
        with col1:
            restecg = st.text_input("Resting electrocardiographic (0,1,2)")
        with col2:
            thalach = st.text_input("Maximum heart rate")
        with col3:
            exang = st.text_input("Exercise induced angina value")
        with col1:
            oldpeak = st.text_input("Oldpeak ")
        with col2:
            slope = st.text_input("Slope of the peak exercise ")
        with col3:
            ca = st.text_input("Number of major vessels (0-3)")
        with col1:
            thal = st.text_input("Thal: 0, 1 or 2")
        
        heart_diagnosis = ' '
        
        if st.button("Heart Disease Test result"):
            try:
                features = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
                features = [float(feature) for feature in features]

                heart_pred = heart_model.predict([features])

                if heart_pred == 1:
                    heart_diagnosis = "The person is more likely affected by heart disease"
                else:
                    heart_diagnosis = "The person shows low probability for having heart disease"

            except Exception as e:
                st.error(f"Error predicting heart disease: {e}")
                heart_diagnosis = "Prediction error"

        st.success(heart_diagnosis)
        
        st.header('Attention! Follow the values given in the range please!') 
        st.markdown("""
            * **Age**
            * **Sex**: 0 represents Male and 1 represents Female
            * **Chest pain type**: [0-1-2-3]
            * **Resting blood pressure**: range [94-200]
            * **Serum cholesterol**: in mg/dl, range: [126-564]
            * **Fasting blood sugar** > 120 mg/dl Normal: 99 milligrams per deciliter (mg/dL) or lower,
                Prediabetes: 100 to 125 mg/dL,
                Diabetes: 126 mg/dL or higher.
                These levels represent the amount of sugar in your blood after fasting (typically not eating for at least 8 hours).
            * **Resting electrocardiographic results** (values 0,1,2)
            * **Maximum heart rate achieved**: range [71-202]
            * **Exercise induced angina**: Binary value, range: 0 or 1
            * **Oldpeak** = Depression induced by exercise relative to rest, range: [0-6.2]
            * **The slope of the peak exercise ST segment**: range: [0-1-2]
            * **Number of major vessels**: range [0-3] colored by fluoroscopy
            * **Thal**: 0 = normal; 1 = fixed defect; 2 = reversible defect
        """)

    ### Parkinson Disease ---------------------------------------------     
    if selected == 'Parkinson Disease Prediction':
        st.title(":person_frowning: Parkinson Disease Prediction ") # Page title
        st.subheader("Please read the Medical descriptions below before entering the values :exclamation:")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            fo = st.text_input("MDVP:Fo(Hz)")
        with col2:
            fhi = st.text_input("MDVP:Fhi(Hz)")
        with col3:
            flo = st.text_input("MDVP:Flo(Hz)")
        with col4:
            Jitter_percent = st.text_input("MDVP:Jitter(%)")
        with col5:
            Jitter_Abs = st.text_input("MDVP:Jitter(Abs)")
        with col1:
            RAP = st.text_input("MDVP:RAP")
        with col2:
            PPQ = st.text_input("MDVP:PPQ")
        with col3:
            DDP = st.text_input("Jitter:DDP")
        with col4:
            Shimmer = st.text_input("MDVP:Shimmer")
        with col5:
            Shimmer_dB = st.text_input("MDVP:Shimmer(dB)")
        with col1:
            APQ3 = st.text_input("Shimmer:APQ3")
        with col2:
            APQ5 = st.text_input("Shimmer:APQ5")
        with col3:
            APQ = st.text_input("MDVP:APQ")
        with col4:
            DDA = st.text_input("Shimmer:DDA")
        with col5:
            NHR = st.text_input("NHR")
        with col1:
            HNR = st.text_input("HNR")
        with col2:
            RPDE = st.text_input("RPDE")
        with col3:
            DFA = st.text_input("DFA")
        with col4:
            spread1 = st.text_input("spread1")
        with col5:
            spread2 = st.text_input("spread2")
        with col1:
            D2 = st.text_input("D2")
        with col2:
            PPE = st.text_input("PPE")
        
        parkinson_diagnosis = ''
        
        if st.button("Parkinson Test Result"):
            try:
                features = [fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ, DDP, Shimmer, Shimmer_dB, APQ3, APQ5, APQ, DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]
                features = [float(feature) for feature in features]

                parkinson_pred = parkinson_model.predict([features])

                if parkinson_pred == 1:
                    parkinson_diagnosis = "The person is more likely affected by Parkinson disease"
                else:
                    parkinson_diagnosis = "The person shows low probability for having Parkinson disease"

            except Exception as e:
                st.error(f"Error predicting Parkinson disease: {e}")
                parkinson_diagnosis = "Prediction error"

        st.success(parkinson_diagnosis)
        
        st.header('Attention! Follow the values given in the range please!') 
        st.markdown("""
            * **MDVP:Fo(Hz)**:  Frequency, average vocal fundamental frequency 
            * **MDVP:Fhi(Hz)**: Frequency, maximum vocal fundamental frequency 
            * **MDVP:Flo(Hz)**: Frequency, minimum vocal fundamental frequency
            * **MDVP:Jitter(%),MDVP:Jitter(Abs),MDVP:RAP,MDVP:PPQ,Jitter:DDP**: Several measures of variation in fundamental frequency 
            * **MDVP:Shimmer,MDVP:Shimmer(dB),Shimmer:APQ3,Shimmer:APQ5,MDVP:APQ,Shimmer:DDA**: Several measures of variation in amplitude 
            * **NHR,HNR**: Two measures of ratio of noise to tonal components in the voice 
            * **RPDE,D2**: Two nonlinear dynamical complexity measures 
            * **DFA**: Signal fractal scaling exponent 
            * **spread1,spread2,PPE**: Three nonlinear measures of fundamental frequency variation
        """)

    ### Breast Cancer Disease -----------------------------------------
    if selected == 'Breast Cancer Prediction':
        st.title(":syringe: Breast Cancer Prediction ") # Page title
        st.subheader("Please read the Medical descriptions below before entering the values :exclamation:")
        
        # User inputs
        col1, col2, col3 = st.columns(3)
        
        with col1:
            mean_radius = st.text_input("Mean radius")
        with col2:
            mean_texture = st.text_input("Mean texture")
        with col3:
            mean_perimeter = st.text_input("Mean perimeter")
        with col1:
            mean_area = st.text_input("Mean area")
        with col2:
            mean_smoothness = st.text_input("Mean smoothness")
        
        breast_cancer_diagnosis = ''
        
        if st.button("Breast Cancer Test Result"):
            try:
                features = [mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness]
                features = [float(feature) for feature in features]

                breast_cancer_pred = breast_model.predict([features])

                if breast_cancer_pred == 1:
                    breast_cancer_diagnosis = "The breast cancer is Malignant"
                else:
                    breast_cancer_diagnosis = "The breast cancer is Benign"

            except Exception as e:
                st.error(f"Error predicting breast cancer: {e}")
                breast_cancer_diagnosis = "Prediction error"

        st.success(breast_cancer_diagnosis)
        
        st.header('Attention! Follow the values given in the range please!') 
        st.markdown("""
            * **Mean Radius**
            * **Mean Texture**
            * **Mean Perimeter**
            * **Mean Area**
            * **Mean Smoothness**
        """)
        
except Exception as e:
    logging.error(f'Error loading models: {e}')
    st.error(f'Error loading models: {e}')
