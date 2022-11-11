import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

eda_data=pd.read_csv("C:/Users/emirh/Downloads/full_data.csv")
eda_data.columns = eda_data.columns.str.lower()

#Binning Categorical Data
age_bin = [[0.0,11.9,17.9,29.9,39.9,49.9,59.9,69.9,79.9,np.inf],["0s","10s","20s","30s","40s","50s","60s","70s","80s+"]]
bmi_bin = [[-np.inf,18.4,24.9,29.9,np.inf],["Underweight","Healthy Weight","Overweight","Obese"]]
glucose_bin = [[-np.inf,114,140,np.inf],["Normal","Prediabetes","Diabetes"]]

def cat_binning(df,var,bin):

    df[var +"_bin"] = pd.cut(x=df[var],bins=bin[0],labels=bin[1])
    return

cat_binning(eda_data,"age",age_bin)
cat_binning(eda_data,"bmi",bmi_bin)
cat_binning(eda_data,"avg_glucose_level",glucose_bin)    

#Alluvial Diagram
gender_dim = go.parcats.Dimension(values=eda_data.gender, label="Gender")
smoke_dim = go.parcats.Dimension(values=eda_data.smoking_status,label="Smoking",categoryarray=["formerly smoked","smokes","never smoked","Unknown"])
bmi_dim = go.parcats.Dimension(values=eda_data.bmi_bin,label="BMI",categoryarray=["Obese","Overweight","Healthy Weight","Underweight"]) 
work_dim = go.parcats.Dimension(values=eda_data.work_type,label="Work Type")    
res_dim = go.parcats.Dimension(values=eda_data.residence_type,label="Residence Type")
age_dim = go.parcats.Dimension(values=eda_data.age_bin,label="Age",categoryorder="category descending")
glc_dim = go.parcats.Dimension(values=eda_data.avg_glucose_level_bin,label="Glucose Level",categoryarray=["Diabetes","Prediabetes","Normal"])
hyp_dim = go.parcats.Dimension(values=eda_data.hypertension,label="Hypertension",categoryarray=[0,1],ticktext=["no","yes"])
heart_dim= go.parcats.Dimension(values=eda_data.heart_disease,label="Heart Disease",categoryarray=[0,1],ticktext=["no","yes"])
str_dim = go.parcats.Dimension(values=eda_data.stroke,categoryarray=[0,1],label="Stroke",ticktext=["not stroke","stroke"])
color = eda_data.stroke
colorscale = [[0, "#6fe396"], [1, "#c2100a"]]

def alluvial_diagram(dimensions_list,title):
    if len(dimensions_list) >1:
        fig = go.Figure(data = [go.Parcats(dimensions=dimensions_list,
        line={"color": color,"colorscale": colorscale,'shape': 'hspline'},
        hoveron="color", hoverinfo="count+probability",
        labelfont={"size": 16, "family": "Times"},
        tickfont={"size": 14, "family": "Times"},
        arrangement="freeform")])

        fig.update_layout(title_text=title,title_font_family='Times',title_font_size=20)      
        
        fig.show()
    else:
        raise ValueError("dimension_list must contain at least 2 dimension. For example [age_dim,str_dim]")
    
alluvial_diagram([smoke_dim,str_dim],"The Patient's Flow Between Smoking Status and Having a Stroke")
alluvial_diagram([glc_dim,str_dim],"The Patient's Flow Between Glucose and Having a Stroke")
alluvial_diagram([bmi_dim,str_dim],"The Patient's Flow Between BMI and Having a Stroke")
alluvial_diagram([bmi_dim,glc_dim,str_dim],"Stroke Patient Flow")
alluvial_diagram([age_dim,smoke_dim,bmi_dim,glc_dim,str_dim],"Stroke Patient Flow")

#Bivariate Analysis
def bivariate_plot(df,var1,var2,hue,title):
    dg = df.groupby([var1,hue])[var2].sum().reset_index()
    sns.set_style("whitegrid", {"font.family":"serif", "font.serif":"Times New Roman"})
    ax = sns.barplot(x=var1, y=var2, hue=hue, data=dg, palette="Set1")
    sns.despine(bottom = True, left = True)
    ax.set(ylabel=None)
    ax.set(xlabel=None)
    ax.set_title(title, loc="left", fontsize=16, fontweight="bold")
    plt.legend(bbox_to_anchor=(1.02, 0.15),loc="best",borderaxespad=0)
    return plt.show()  

bivariate_plot(eda_data,"bmi_bin","stroke","gender","Patients with Stroke by BMI")
bivariate_plot(eda_data,"avg_glucose_level_bin","stroke","gender","Patients with Stroke by Glucose Level")
bivariate_plot(eda_data,"smoking_status","stroke","gender","Patients with Stroke by Smoking Status")
bivariate_plot(eda_data,"age_bin","stroke","gender","Patients with Stroke by Age")
bivariate_plot(eda_data,"work_type","stroke","gender","Patients with Stroke by Work Type")
bivariate_plot(eda_data,"age_bin","heart_disease","gender","Patients with Heart Disease by Age")
bivariate_plot(eda_data,"age_bin","hypertension","gender","Patients with Hypertension by Age")
bivariate_plot(eda_data,"residence_type","stroke","gender","Patients with Stroke by Residence Type")
bivariate_plot(eda_data,"ever_married","stroke","gender","Patients with Stroke by Marriage Status")