
import streamlit as st
import numpy as np
import pandas as pd
from matplotlib.image import imread
import matplotlib.pyplot as plt
import seaborn as sns
import requests


########################################################
# Session for the API
########################################################
def fetch(session, url):

    try:
        result = session.get(url)
        return result.json()
    except Exception:
        return {}

session = requests.Session()

########################################################
# Functions to call the EndPoints
########################################################

def client():
    #Getting Client details
    response = fetch(session, f"http://127.0.0.1:8000/api/clients")
    if response:
        return response["clientsId"]
    else:
        return "Error"
    
def client_details(id):
    #Getting Client details
    response = fetch(session,f"http://127.0.0.1:8000/api/clients/{id}")
    if response:
        return response
    else:
        return "Error"
    
def client_prediction(id):
    response = fetch(session, f"http://127.0.0.1:8000/api/clients/{id}/prediction")
    if response:
        return response
    else:
        return "Error"    
    
def chart_kde(title,row,df,col,client):
    """Définition des graphes KDE avec une ligne verticale indiquant la position du client"""
    with row:
        st.subheader(title)
        fig,ax = plt.subplots()
        sns.kdeplot(df.loc[df["TARGET"]==0,col],color="green", label = "Target == 0")
        sns.kdeplot(df.loc[df["TARGET"]==1,col],color="red", label = "Target == 1")
        plt.axvline(x=df.iloc[client, df.columns.get_loc(col)],ymax=0.95,color="black")
        plt.legend()
        st.pyplot(fig)

def chart_bar(title,row,df,col,client):
    """Définition des graphes barres avec une ligne horizontale indiquant la position du client"""
    with row:
        st.subheader(title)
        fig,ax = plt.subplots()
        data=df[["TARGET",col]]
        if data[col].dtypes!="object":
            data[col]=data[col].astype("str")

            data1=round(data[col].loc[data["TARGET"]==1].value_counts()/data[col].loc[data["TARGET"]==1].value_counts().sum()*100,2)
            data0=round(data[col].loc[data["TARGET"]==0].value_counts()/data[col].loc[data["TARGET"]==0].value_counts().sum()*100,2)
            data=pd.concat([pd.DataFrame({"Pourcentage":data0,"TARGET":0}),pd.DataFrame({"Pourcentage":data1,"TARGET":1})]).reset_index().rename(columns={"index":col})
            sns.barplot(data=data,x="Pourcentage", y=col, hue="TARGET", palette=["green","red"], order=sorted(data[col].unique()));
            
            data[col]=data[col].astype("int64")

            plt.axhline(y=sorted(data[col].unique()).index(df.loc[client,col]),xmax=0.95,color="black",linewidth=4)
            st.pyplot(fig)
        else:

            data1=round(data[col].loc[data["TARGET"]==1].value_counts()/data[col].loc[data["TARGET"]==1].value_counts().sum()*100,2)
            data0=round(data[col].loc[data["TARGET"]==0].value_counts()/data[col].loc[data["TARGET"]==0].value_counts().sum()*100,2)
            data=pd.concat([pd.DataFrame({"Pourcentage":data0,"TARGET":0}),pd.DataFrame({"Pourcentage":data1,"TARGET":1})]).reset_index().rename(columns={"index":col})
            sns.barplot(data=data,x="Pourcentage", y=col, hue="TARGET", palette=["green","red"], order=sorted(data[col].unique()));
            
            plt.axhline(y=sorted(data[col].unique()).index(df.loc[client,col]),xmax=0.95,color="black",linewidth=4)
            st.pyplot(fig)

def display_charts(df,client):
	"""Affichage des graphes de comparaison pour le client sélectionné """
	row1_1,row1_2,row1_3 = st.columns(3)
	st.write('')
	row2_10,row2_2,row2_3 = st.columns(3)
	
	chart_kde("Répartition de l'age",row1_1,df,'AGE_INT',client)
	chart_kde("Répartition des revenus",row1_2,df,'AMT_INCOME_TOTAL',client)
	chart_bar("Répartition du nombre d'enfants",row1_3,df,'CNT_CHILDREN',client)

	chart_bar("Répartition du statut professionel",row2_10,df,'NAME_INCOME_TYPE',client)
	chart_bar("Répartition du niveau d'études",row2_2,df,'NAME_EDUCATION_TYPE',client)
	chart_bar("Répartition du type de logement",row2_3,df,'NAME_HOUSING_TYPE',client)
  
def app():
    """Fonction générant la page 2 du dashboard. Ne prend pas de paramètre en entrée.
    """
    #Chargement des données nécessaires au dashboard
    st.title("Comparaison clientèle")    
    logo = imread("./app_pages/logo.png")

    st.sidebar.image(logo)
    st.sidebar.write("")
    st.sidebar.write("")
  
    #Get Client
    client_id = st.sidebar.selectbox("Client Id List", client())
    
    #Get Prediction for selected client
    prediction = client_prediction(client_id)
    prediction = pd.read_json(prediction)
    
    #Infos Client
    st.sidebar.markdown("ID client: " + str(client_id))
    
    st.sidebar.markdown("Sexe: " + prediction["CODE_GENDER"].iloc[0])
    st.sidebar.markdown("Statut familial: " + prediction["NAME_FAMILY_STATUS"].iloc[0])
    st.sidebar.markdown("Enfants: " + str(prediction["CNT_CHILDREN"].iloc[0].astype("int64")))
    st.sidebar.markdown("Age: " + str(prediction["AGE_INT"].iloc[0].astype("int64")))
    st.sidebar.markdown("Statut pro.: " + prediction["NAME_INCOME_TYPE"].iloc[0])
    st.sidebar.markdown("Niveau d'études: " + prediction["NAME_EDUCATION_TYPE"].iloc[0])
    
    #Data comparison (training set)
    df = pd.read_csv("./dashboard_data/df_train.csv")
    
    #changing type of features
    df["CNT_CHILDREN"] = df["CNT_CHILDREN"].astype("int64")
    df["AGE_INT"] = df["AGE_INT"].astype("int64")
    
    #renaming pred column in prediction row to TARGET to match with training set
    prediction.rename(columns={"pred": "TARGET"})
    
    #concatenate training set and prediction row
    frames = [prediction, df]
    df = pd.concat(frames)
    
    #Reset index
    df = df.reset_index(drop=True)
    
    #display charts
    idx_client = df.index[df["SK_ID_CURR"]==client_id][0]
    display_charts(df, idx_client)
 
if __name__ == "__main__":
    app()