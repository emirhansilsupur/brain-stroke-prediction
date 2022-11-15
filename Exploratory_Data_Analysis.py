import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st


def get_data():
    eda_data = pd.read_csv("full_data.csv")
    eda_data.columns = eda_data.columns.str.lower()
    return eda_data


data = get_data()

# Binning Categorical Data
age_bin = [
    [0.0, 11.9, 17.9, 29.9, 39.9, 49.9, 59.9, 69.9, 79.9, np.inf],
    ["0s", "10s", "20s", "30s", "40s", "50s", "60s", "70s", "80s+"],
]
bmi_bin = [
    [-np.inf, 18.4, 24.9, 29.9, np.inf],
    ["Underweight", "Healthy Weight", "Overweight", "Obese"],
]
glucose_bin = [[-np.inf, 114, 140, np.inf], ["Normal", "Prediabetes", "Diabetes"]]


def cat_binning(df, var, bin):

    df[var + "_bin"] = pd.cut(x=df[var], bins=bin[0], labels=bin[1])
    return


cat_binning(data, "age", age_bin)
cat_binning(data, "bmi", bmi_bin)
cat_binning(data, "avg_glucose_level", glucose_bin)

data.rename(
    columns={
        "gender": "Gender",
        "age_bin": "Age",
        "hypertension": "Hypertension",
        "heart_disease": "Heart Disease",
        "ever_married": "Marriage Status",
        "work_type": "Work Type",
        "residence_type": "Residence Type",
        "avg_glucose_level_bin": "Glucose Level",
        "bmi_bin": "BMI",
        "smoking_status": "Smoking Status",
        "stroke": "Stroke",
    },
    inplace=True,
)

# Alluvial Diagram
def get_parcats():
    gender_dim = go.parcats.Dimension(values=data.Gender, label="Gender")
    smoke_dim = go.parcats.Dimension(
        values=data["Smoking Status"],
        label="Smoking",
        categoryarray=["formerly smoked", "smokes", "never smoked", "Unknown"],
    )
    bmi_dim = go.parcats.Dimension(
        values=data.BMI,
        label="BMI",
        categoryarray=["Obese", "Overweight", "Healthy Weight", "Underweight"],
    )
    work_dim = go.parcats.Dimension(values=data["Work Type"], label="Work Type")
    res_dim = go.parcats.Dimension(
        values=data["Residence Type"], label="Residence Type"
    )
    age_dim = go.parcats.Dimension(
        values=data.Age, label="Age", categoryorder="category descending"
    )
    glc_dim = go.parcats.Dimension(
        values=data["Glucose Level"],
        label="Glucose Level",
        categoryarray=["Diabetes", "Prediabetes", "Normal"],
    )
    hyp_dim = go.parcats.Dimension(
        values=data.Hypertension,
        label="Hypertension",
        categoryarray=[0, 1],
        ticktext=["no", "yes"],
    )
    heart_dim = go.parcats.Dimension(
        values=data["Heart Disease"],
        label="Heart Disease",
        categoryarray=[0, 1],
        ticktext=["no", "yes"],
    )
    
    married_dim = go.parcats.Dimension(values=data["Marriage Status"],label="Marriage Status")
    
    str_dim = go.parcats.Dimension(
        values=data.Stroke,
        categoryarray=[0, 1],
        label="Stroke",
        ticktext=["not stroke", "stroke"],
    )
    color = data.Stroke
    colorscale = [[0, "#6fe396"], [1, "#c2100a"]]
    return (
        gender_dim,
        smoke_dim,
        bmi_dim,
        work_dim,
        res_dim,
        age_dim,
        glc_dim,
        hyp_dim,
        heart_dim,
        married_dim,
        str_dim,
        color,
        colorscale,
    )


(
    gender_dim,
    smoke_dim,
    bmi_dim,
    work_dim,
    res_dim,
    age_dim,
    glc_dim,
    hyp_dim,
    heart_dim,
    married_dim,
    str_dim,
    color,
    colorscale,
) = get_parcats()


def alluvial_diagram(dimensions_list):

    fig = go.Figure(
            data=[
                go.Parcats(
                    dimensions=dimensions_list,
                    line={"color": color, "colorscale": colorscale, "shape": "hspline"},
                    hoveron="color",
                    hoverinfo="count+probability",
                    labelfont={"size": 16, "family": "Times"},
                    tickfont={"size": 14, "family": "Times"},
                    arrangement="freeform",
                )
            ]
        )


    st.plotly_chart(fig, use_container_width=True)



# Bivariate Analysis
def bivariate_plot(var1, var2, hue):
    dg = data.groupby([var1, hue])[var2].sum().reset_index()
    sns.set_style("whitegrid", {"font.family": "serif", "font.serif": "Times New Roman"})
    ax = sns.barplot(x=var1, y=var2, hue=hue, data=dg, palette="Set1")
    sns.despine(bottom=True, left=True)
    ax.set(ylabel=var2)
    ax.set(xlabel=var1)
    plt.legend(bbox_to_anchor=(1.02, 0.15), loc="best", borderaxespad=0)
    return st.pyplot(plt)


