from Data_Operation_and_Model_Building import *
from Exploratory_Data_Analysis import *
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

model = joblib.load("lgb_model.pkl")


st.set_page_config(page_title="Brain Stroke Prediction")

tabs=["Prediction","Visualization","About Me"]

page=st.sidebar.radio("Tabs",tabs)


if page =="Prediction":
    st.markdown("<h1 style='text-align:center;'>Brain Stroke Prediction</h1>",unsafe_allow_html=True)
    st.markdown("<h2 style='text-align:center;'>Personal Information</h2>",unsafe_allow_html=True)
    gender_ = st.radio("Gender",["Female","Male"])
    age_ = st.slider("Age",0,100)
    work_ = st.radio("Work Type",["Children","Private","Self-Employed","Government Job"])
    ever_married_ = st.radio("Did you ever get married?",["Yes","No"])
    residence_ = st.radio("What is your residence type?",["Urban","Rural"])

    st.markdown("<h2 style='text-align:center;'>Health Information</h2>",unsafe_allow_html=True)
    smoking_ = st.radio("What is your smoking type?",["Smoker","Formerly smoker","Never smoked","Unknown"])
    if st.checkbox("Dont Know BMI? Use height and weight"):
        height = st.slider("Enter User's Height in cm",value=100.,step=1.,format="%.2f")
        weight = st.slider("Enter User's Weight in kgs",value=100.,step=1.,format="%.2f")
        bmi_ = weight / (height/100)**2
        st.write("BMI of user is {:.2f} and will be autoupdated".format(bmi_))
    else:
        bmi_ = st.slider("Enter User's BMI",min_value=10.,max_value=50.,step=0.01,format="%.2f")
    #bmi_ = st.number_input("BMI",step=1.,format="%.2f")
    glucose_level_ = st.slider("Glucose Level(eAG)",min_value=0.,max_value=300.,step=1.,format="%.2f")
    heart_ = st.radio("Do you have heart disease?",["Yes","No"])
    hypertension_ = st.radio("Do you have hypertension?",["Yes","No"])
    button=st.button("Click me")
    
    

    cols=["gender","age","hypertension","heart_disease","ever_married","residence_type","bmi","avg_glucose_level","work_type", "smoking_status"]

    row = np.array([gender_,age_,hypertension_,heart_,ever_married_,residence_,bmi_,glucose_level_,work_,smoking_])
    X = pd.DataFrame(data=[row],columns=cols)
    X["age"]=X["age"].astype("float")
    X["avg_glucose_level"] = X["avg_glucose_level"].astype("float")
    X["bmi"] = X["bmi"].astype("float")
    
  
    X["gender"] = [1 if i == "Male" else 0 for i in X["gender"]]
    X["hypertension"] = [1 if i == "Yes" else 0 for i in X["hypertension"]]
    X["heart_disease"] = [1 if i == "Yes" else 0 for i in X["heart_disease"]]
    X["ever_married"] = [1 if i == "Yes" else 0 for i in X["ever_married"]]
    X["residence_type"] = [1 if i == "Urban" else 0 for i in X["residence_type"]]
    
    if [X["work_type"] == "Private"]:
        X["work_type_private"]= 1
        X["work_type_self_employed"]=0
        X["work_type_children"]=0
        X.drop(columns="work_type",inplace=True)

    elif [X["work_type"] == "Children"]:
        X["work_type_private"]=0
        X["work_type_self_employed"]=0
        X["work_type_chilren"]= 1
        X.drop(columns="work_type",inplace=True)
                
    elif [X["work_type"] == "Self-Employed"]:
        X["work_type_private"]=0
        X["work_type_self_employed"]=1
        X["work_type_children"]= 0
        X.drop(columns="work_type",inplace=True)

    elif [X["work_type"] == "Government Job"]:
        X["work_type_private"]= 0
        X["work_type_self_employed"]=0
        X["work_type_children"]=0
        X.drop(columns="work_type",inplace=True)


    if [X["smoking_status"] == "Formerly smoker"]:
        X["smoking_status_formerly_smoker"] = 1
        X["smoking_status_never_smoked"] = 0
        X["smoking_status_smokes"] = 0
        X.drop(columns="smoking_status",inplace=True)

    elif [X["smoking_status"] == "Smoker"]:
        X["smoking_status_formerly_smoker"] = 0
        X["smoking_status_never_smoked"] = 0
        X["smoking_status_smokes"] = 1
        X.drop(columns="smoking_status",inplace=True)


    elif [X["smoking_status"] == "Never smoked"]:
        X["smoking_status_formerly_smoker"] = 0
        X["smoking_status_never_smoked"] = 1
        X["smoking_status_smoker"] = 0   
        X.drop(columns="smoking_status",inplace=True)
  
    
    if button==True:
        X = X[["gender","age","hypertension","heart_disease","ever_married","residence_type","bmi","avg_glucose_level","work_type_private","work_type_self_employed",
        "work_type_children","smoking_status_formerly_smoker","smoking_status_never_smoked","smoking_status_smokes"]]
    
        #st.write.type(gender_)
        prediction = model.predict(X)

        #Prediction Probability
        pred_prob = model.predict_proba(X)
        stroke_prob = pred_prob[0][1]*100
        if prediction==1:
            st.error("You have Higher Chances of having a Stroke😔")
        else:
            st.success("You have Lower Chances of having a Stroke😊")    

        if stroke_prob < 25:
            st.success(f"Probability of Occurance of Stroke is %{stroke_prob}")
        elif stroke_prob < 50:
            st.info(f"Probability of Occurance of Stroke is %{stroke_prob}")
        elif stroke_prob < 75:
            st.warning(f"Probability of Occurance of Stroke is %{stroke_prob}")
        else:
            st.error(f"Probability of Occurance of Stroke is %{stroke_prob}")
       
        
    
