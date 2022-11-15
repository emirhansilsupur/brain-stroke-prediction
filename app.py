from Data_Operation_and_Model_Building import *
from Exploratory_Data_Analysis import *
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import time
import warnings

warnings.filterwarnings("ignore")

model = joblib.load("lgb_model.pkl")


st.set_page_config(page_title="Brain Stroke Prediction")

tabs = ["Prediction", "Visualization", "About"]
page = st.sidebar.radio("Tabs", tabs)

if page == "Prediction":
    st.markdown(
        "<h1 style='text-align:center;'>Brain Stroke Prediction</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<h2 style='text-align:center;'>Personal Information</h2>",
        unsafe_allow_html=True,
    )
    gender_ = st.radio("Gender", ["Female", "Male"])
    age_ = st.slider("Age", 0, 100)
    work_ = st.radio(
        "Work Type", ["Children", "Private", "Self-Employed", "Government Job"]
    )
    ever_married_ = st.radio("Did you ever get married?", ["Yes", "No"])
    residence_ = st.radio("What is your residence type?", ["Urban", "Rural"])

    st.markdown(
        "<h2 style='text-align:center;'>Health Information</h2>", unsafe_allow_html=True
    )
    smoking_ = st.radio(
        "What is your smoking type?",
        ["Smoker", "Formerly smoker", "Never smoked", "Unknown"],
    )

    if st.checkbox("Dont Know BMI?"):
        height = st.slider(
            "Enter User's Height in cm", value=200.0, step=1.0, format="%.2f"
        )
        weight = st.slider(
            "Enter User's Weight in kgs", value=200.0, step=1.0, format="%.2f"
        )
        bmi_ = weight / (height / 100) ** 2
        st.write(f"BMI of user is {bmi_} and will be autoupdated")
    else:
        bmi_ = st.slider(
            "Enter User's BMI", min_value=10.0, max_value=50.0, step=0.01, format="%.2f"
        )

    glucose_level_ = st.slider(
        "Glucose Level(eAG)", min_value=50.0, max_value=300.0, step=1.0, format="%.2f"
    )
    heart_ = st.radio("Do you have heart disease?", ["Yes", "No"])
    hypertension_ = st.radio("Do you have hypertension?", ["Yes", "No"])
    button = st.button("Predict")

    cols = [
        "gender",
        "age",
        "hypertension",
        "heart_disease",
        "ever_married",
        "residence_type",
        "bmi",
        "avg_glucose_level",
        "work_type",
        "smoking_status",
    ]

    row = np.array(
        [
            gender_,
            age_,
            hypertension_,
            heart_,
            ever_married_,
            residence_,
            bmi_,
            glucose_level_,
            work_,
            smoking_,
        ]
    )
    X = pd.DataFrame(data=[row], columns=cols)
    X["age"] = X["age"].astype("float")
    X["avg_glucose_level"] = X["avg_glucose_level"].astype("float")
    X["bmi"] = X["bmi"].astype("float")

    X["gender"] = [1 if i == "Male" else 0 for i in X["gender"]]
    X["hypertension"] = [1 if i == "Yes" else 0 for i in X["hypertension"]]
    X["heart_disease"] = [1 if i == "Yes" else 0 for i in X["heart_disease"]]
    X["ever_married"] = [1 if i == "Yes" else 0 for i in X["ever_married"]]
    X["residence_type"] = [1 if i == "Urban" else 0 for i in X["residence_type"]]

    if [X["work_type"] == "Private"]:
        X["work_type_private"] = 1
        X["work_type_self_employed"] = 0
        X["work_type_children"] = 0
        X.drop(columns="work_type", inplace=True)

    elif [X["work_type"] == "Children"]:
        X["work_type_private"] = 0
        X["work_type_self_employed"] = 0
        X["work_type_chilren"] = 1
        X.drop(columns="work_type", inplace=True)

    elif [X["work_type"] == "Self-Employed"]:
        X["work_type_private"] = 0
        X["work_type_self_employed"] = 1
        X["work_type_children"] = 0
        X.drop(columns="work_type", inplace=True)

    elif [X["work_type"] == "Government Job"]:
        X["work_type_private"] = 0
        X["work_type_self_employed"] = 0
        X["work_type_children"] = 0
        X.drop(columns="work_type", inplace=True)

    if [X["smoking_status"] == "Formerly smoker"]:
        X["smoking_status_formerly_smoker"] = 1
        X["smoking_status_never_smoked"] = 0
        X["smoking_status_smokes"] = 0
        X.drop(columns="smoking_status", inplace=True)

    elif [X["smoking_status"] == "Smoker"]:
        X["smoking_status_formerly_smoker"] = 0
        X["smoking_status_never_smoked"] = 0
        X["smoking_status_smokes"] = 1
        X.drop(columns="smoking_status", inplace=True)

    elif [X["smoking_status"] == "Never smoked"]:
        X["smoking_status_formerly_smoker"] = 0
        X["smoking_status_never_smoked"] = 1
        X["smoking_status_smoker"] = 0
        X.drop(columns="smoking_status", inplace=True)

    if button == True:
        info = st.info("Please wait, prediction is in progress:hourglass:")
        progress_bar = st.progress(0)
        for perc_completed in range(100):
            time.sleep(0.05)
            progress_bar.progress(perc_completed + 1)
        info.empty()

        X = X[
            [
                "gender",
                "age",
                "hypertension",
                "heart_disease",
                "ever_married",
                "residence_type",
                "bmi",
                "avg_glucose_level",
                "work_type_private",
                "work_type_self_employed",
                "work_type_children",
                "smoking_status_formerly_smoker",
                "smoking_status_never_smoked",
                "smoking_status_smokes",
            ]
        ]

        prediction = model.predict(X)
        pred_prob = model.predict_proba(X)
        stroke_prob = pred_prob[0][1] * 100

        if prediction == 1:
            st.error("You have higher chances of having a stroke:disappointed:")
        else:
            st.success("You have lower chances of having a stroke🥳")

        if stroke_prob < 1:
            st.success(
                f"Probability of occurance of stroke is %{round(stroke_prob,2)}:blush:"
            )

        elif stroke_prob < 25:
            st.success(
                f"Probability of occurance of stroke is %{round(stroke_prob,2)}:smirk:"
            )
        elif stroke_prob < 50:
            st.info(
                f"Probability of occurance of stroke is %{round(stroke_prob,2)}:hushed:"
            )
        elif stroke_prob < 75:
            st.warning(
                f"Probability of occurance of stroke is %{round(stroke_prob,2)}:worried:"
            )
        else:
            st.error(
                f"Probability of occurance of Stroke is %{round(stroke_prob,2)}:fearful:"
            )

if page == "Visualization":
    col1, col2 = st.columns([1, 8])
    col2.header("Patients with Stroke by Gender")

    b_x = st.selectbox(
        "Choose a 'x' variable",
        [
            "BMI",
            "Glucose Level",
            "Smoking Status",
            "Age",
            "Work Type",
            "Heart Disease",
            "Hypertension",
            "Residence Type",
            "Marriage Status",
        ],
    )

    b_y = st.selectbox(
        "Choose a 'y' variable", ["Stroke", "Heart Disease", "Hypertension"]
    )

    b_hue = st.selectbox(
        "Choose a 'hue' variable",
        [
            "Gender",
            "Marriage Status",
            "Residence Type",
            "Heart Disease",
            "Hypertension",
        ],
    )

    if b_x == b_y:
        st.error("Please select another 'y' variable")

    elif b_y == b_hue:
        st.error("Please select another 'hue' variable")
    elif b_x == b_hue:
        st.error("Please select another 'hue' variable")

    else:

        st.write("-----" * 34)
        bivariate_plot(b_x, b_y, b_hue)
        st.write("-----" * 34)

    ####
    col1, col2 = st.columns([1, 4])
    col2.header("Flow of Stroke Patients")
    x_axis = st.multiselect(
        "You can view the flow by selecting multiple variables",
        [
            "Gender",
            "BMI",
            "Glucose Level",
            "Smoking Status",
            "Age",
            "Work Type",
            "Residence Type",
            "Marriage Status",
        ],
    )

    all_parcats = []

    for i in x_axis:

        if i == "Smoking Status":
            all_parcats.append(smoke_dim)

        elif i == "Gender":
            all_parcats.append(gender_dim)

        elif i == "BMI":
            all_parcats.append(bmi_dim)
        elif i == "Glucose Level":
            all_parcats.append(glc_dim)
        elif i == "Age":

            all_parcats.append(age_dim)

        elif i == "Heart Disease":
            all_parcats.append(heart_dim)

        elif i == "Hypertension":

            all_parcats.append(hyp_dim)

        elif i == "Work Type":
            all_parcats.append(work_dim)

        elif i == "Residence Type":
            all_parcats.append(res_dim)
        else:

            all_parcats.append(married_dim)

    all_parcats.insert(len(all_parcats), str_dim)
    st.write("-----" * 34)
    alluvial_diagram(all_parcats)
    st.write("-----" * 34)


if page == "About":
    st.header(":mailbox:Get In Touch With Me")
    st.markdown("""**[Linkedin](https://www.linkedin.com/in/emirhansilsupur/)**""")
    st.markdown("""**[Github](https://github.com/emirhansilsupur)**""")
    st.markdown("""**[Kaggle](https://www.kaggle.com/emirslspr)**""")
    st.markdown(
        """**[Full Code](https://github.com/emirhansilsupur/brain-stroke-prediction)**"""
    )

    st.markdown(
        """**[Click For More Information About The Project](https://www.kaggle.com/datasets/zzettrkalpakbal/full-filled-brain-stroke-dataset)**"""
    )

    st.caption("*This project was developed to learn the model deployment.*")
