import pickle
import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu
import sklearn
import xgboost
# App title
st.set_page_config(page_title="MY FITNESS APP")
Hmodel = pickle.load(open('heart_model.sav','rb'))
Bmodel = pickle.load(open('Breast_Cancer_model.sav','rb'))
Dmodel = pickle.load(open('Diabetes_model.sav','rb'))
with st.sidebar:
    st.title("My Fitness App")
    selected = option_menu('Check Your Health',['Heart Disease Prediction','Breast Cancer Prediction','Diabetes Prediction'],default_index = 0,icons = ['heart','person','activity'])
if selected=='Heart Disease Prediction':
    st.title('Heart Disease Prediction')
    age = st.number_input('Age')
    sex = st.number_input('Sex')
    cp = st.number_input('Chest Pain')
    trestbps = st.number_input('Resting Blood Pressure')
    chol = st.number_input('Cholesterol')
    fbs = st.number_input('Fasting Blood Sugar')
    restecg = st.number_input('Resting Electrocardiogram')
    thalach = st.number_input('Maximum Heart Rate')
    exang = st.number_input('Exercise-Induced Angina')
    oldpeak = st.number_input('ST Depression Induced by Exercise')
    slope = st.number_input('Slope of the ST Segment During Exercise')
    ca = st.number_input('Number of Major Vessels Colored by Fluoroscopy')
    thal = st.number_input('Thallium Stress Test Result')
    a = ''
    if st.button('Test Result'):
        input=(age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal)
        input_array = np.asarray(input)
        input_reshaped = input_array.reshape(1,-1)
        result = Hmodel.predict(input_reshaped)
        
        if(result[0]==0):
            a = 'Patient do not have a Heart Disease'
        else:
            a = 'Patient have a Heart Disease'
    st.success(a)
elif selected=='Breast Cancer Prediction':
    st.title('Breast Cancer Prediction')
    col1,col2,col3 = st.columns(3)
    with col1:
        mean_radius = st.number_input('mean_radius')
    with col2:
        mean_texture = st.number_input('mean_texture')
    with col3:
        mean_perimeter = st.number_input('mean_perimeter')
    with col1:
        mean_area =st.number_input('mean_area')
    with col2:
        mean_smoothness = st.number_input('mean_smoothness')
    with col3:
        mean_compactness = st.number_input('mean_compactness')
    with col1:
        mean_concavity = st.number_input('mean_concavity')
    with col2:
        mean_concave_points = st.number_input('mean_concave_points')
    with col3:
        mean_symmetry = st.number_input('mean_symmetry')
    with col1:
        mean_fractal_dimension =st.number_input('mean_fractal_dimension')
    with col2:
        radius_error = st.number_input('radius_error')
    with col3:
        texture_error = st.number_input('texture_error')
    with col1:
        perimeter_error = st.number_input('perimeter_error')
    with col2:
        area_error = st.number_input('area_error')
    with col3:
        smoothness_error = st.number_input('smoothness_error')
    with col1:
        compactness_error = st.number_input('compactness_error')
    with col2:
        concavity_error = st.number_input('concavity_error')
    with col3:
        concave_points_error = st.number_input('concave_points_error')
    with col1:
        symmetry_error = st.number_input('symmetry_error')
    with col2:
        fractal_dimension_error = st.number_input('fractal_dimension_error')
    with col3:
        worst_radius = st.number_input('worst_radius')
    with col1:
        worst_texture = st.number_input('worst_texture')
    with col2:
        worst_perimeter = st.number_input('worst_perimeter')
    with col3:
        worst_area = st.number_input('worst_area')
    with col1:
        worst_smoothness = st.number_input('worst_smoothness')
    with col2:
        worst_compactness = st.number_input('worst_compactness')
    with col3:
        worst_concavity = st.number_input('worst_concavity')
    with col1:
        worst_concave_points = st.number_input('worst_concave_points')
    with col2:
        worst_symmetry = st.number_input('worst_symmetry')
    with col3:
        worst_fractal_dimension = st.number_input('worst_fractal_dimension')
    a = ''
    if st.button('Test Result'):
        input = (mean_radius,mean_texture,mean_perimeter,mean_area,mean_smoothness,mean_compactness,mean_concavity,mean_concave_points,mean_symmetry,mean_fractal_dimension,radius_error,texture_error,perimeter_error,area_error,smoothness_error,compactness_error,concavity_error,concave_points_error,symmetry_error,fractal_dimension_error,worst_radius,worst_texture,worst_perimeter,worst_area,worst_smoothness,worst_compactness,worst_concavity,worst_concave_points,worst_symmetry,worst_fractal_dimension)
        input_array = np.asarray(input)
        input_reshaped = input_array.reshape(1,-1)
        result = Bmodel.predict(input_reshaped)
        if result[0]==0:
            a = 'Patient is Malignant'
        else:
            a = 'Patient is Benign'
    st.success(a)
else:
    st.title('Diabetes Prediction')
    Pregnancies = st.number_input('Number of Pregnancies')
    Glucose = st.number_input('Glucose Level')
    BloodPressure = st.number_input('BloodPressure')
    SkinThickness = st.number_input('SkinThickness')
    Insulin = st.number_input('Insulin')
    BMI = st.number_input('BMI')
    DiabetesPedigreeFunction = st.number_input('DiabetesPedigreeFunction')
    Age = st.number_input('Age')
    a = ''
    if st.button('Test Result'):
        result = Dmodel.predict([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])
        if(result[0]==0):
            a = 'You do not have a Diabetes'
        else:
            a = 'You have a Diabetes'
        
    st.success(a)