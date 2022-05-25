# coding=utf-8

import streamlit as st
import numpy as np
import pandas as pd
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report
from matplotlib.image import imread

#######################################################################
# Loading data (labelled)
#---------------------------------------------------------------------

@st.experimental_singleton(suppress_st_warning=True)
def app():
    """Fonction générant la page 1 du dashboard. Ne prend pas de paramètre en entrée.
    """
    logo = imread("./app_pages/logo.png")

    st.sidebar.image(logo)
    st.sidebar.write("")
    st.sidebar.write("")
    
    st.title("Aperçu de la population de prospect (labellisée)")
        
    df = pd.read_csv("./dashboard_data/df_train.csv").astype("object")
    
    #Sample data for Customer profile Analysis    
    colonnes_pandas_profiling = [
                             "CODE_GENDER",
                             "AGE_INT", 
                             "NAME_TYPE_SUITE",
                             "NAME_EDUCATION_TYPE",
                             "NAME_INCOME_TYPE",
                             "ORGANIZATION_TYPE",
                             "OCCUPATION_TYPE",
                             "NAME_HOUSING_TYPE",
                             "CNT_CHILDREN", 
                             "AMT_INCOME_TOTAL", 
                             "AMT_GOODS_PRICE"                             
                             ]
    df_pandas_profiling = df.loc[:,colonnes_pandas_profiling]    

    
    #Pandas Profiling Report
    st.write("Analyse exploratoire d'un échantillon du dataset labellisé de prospect (seules 11 des 101 variables du dataset sont présentées afin de comprendre le profil des prospects)")
    pr = df_pandas_profiling.profile_report(minimal=True)
    st_profile_report(pr)

if __name__ == "__main__":
    app()