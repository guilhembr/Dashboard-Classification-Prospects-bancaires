
import streamlit as st
import numpy as np
import pandas as pd
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report
from matplotlib.image import imread
import matplotlib.pyplot as plt
import seaborn as sns


def chart_kde(title,row,df,col,client):
    """Définition des graphes KDE avec une ligne verticale indiquant la position du client"""
    with row:
        st.subheader(title)
        fig,ax = plt.subplots()
        sns.kdeplot(df.loc[df["TARGET"]==0,col],color="green", label = "Target == 0")
        sns.kdeplot(df.loc[df["TARGET"]==1,col],color="red", label = "Target == 1")
        plt.axvline(x=df.loc[client,col],ymax=0.95,color="black")
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
   
def app():
    """Fonction générant la page 2 du dashboard. Ne prend pas de paramètre en entrée.
    """
    #Chargement des données nécessaires au dashboard
    st.title("Comparaison clientèle")
        
    df = pd.read_csv("/Users/guilhemberthou/dev/P7_Scoring/dashboard_data/df_test.csv")
    logo = imread("/Users/guilhemberthou/dev/P7_Scoring/app_pages/logo.png")

    st.sidebar.image(logo)
    st.sidebar.write("")
    st.sidebar.write("")
  
    #Get Client
    client = st.sidebar.selectbox("Client", df["SK_ID_CURR"])
    idx_client = df.index[df["SK_ID_CURR"]==client][0]
    
    #Infos Client
    st.sidebar.markdown("ID client: " + str(client))
    st.sidebar.markdown("Sexe: " + df.loc[idx_client,"CODE_GENDER"])
    st.sidebar.markdown("Statut familial: " + df.loc[idx_client,"NAME_FAMILY_STATUS"])
    st.sidebar.markdown("Enfants: " + str(df.loc[idx_client,"CNT_CHILDREN"]))
    st.sidebar.markdown("Age: " + str(df.loc[idx_client,"AGE_INT"]))	
    st.sidebar.markdown("Statut pro.: " + df.loc[idx_client,"NAME_INCOME_TYPE"])
    st.sidebar.markdown("Niveau d'études: " + df.loc[idx_client,"NAME_EDUCATION_TYPE"])
    
    #comparaison
    row1_1,row1_2,row1_3 = st.columns(3)
    st.write("")
    row2_1,row2_2,row2_3 = st.columns(3)
 
    chart_kde("Répartition de l'age",row1_1,df,"AGE_INT",client)
    chart_kde("Répartition des revenus",row1_2,df,"AMT_INCOME_TOTAL",client)
    chart_bar("Répartition du nombre d'enfants",row1_3,df,"CNT_CHILDREN",client)

    chart_bar("Répartition du statut professionel",row2_1,df,"NAME_INCOME_TYPE",client)
    chart_bar("Répartition du niveau d'études",row2_2,df,"NAME_EDUCATION_TYPE",client)
    chart_bar("Répartition du type de logement",row2_3,df,"NAME_HOUSING_TYPE",client)
    
    st.dataframe(df[["SK_ID_CURR","CODE_GENDER","AGE_INT","NAME_FAMILY_STATUS","CNT_CHILDREN",
    "NAME_EDUCATION_TYPE","FLAG_OWN_CAR","FLAG_OWN_REALTY","NAME_HOUSING_TYPE",
    "NAME_INCOME_TYPE","AMT_INCOME_TOTAL","AMT_CREDIT","AMT_ANNUITY"]])

if __name__ == "__main__":
    app()