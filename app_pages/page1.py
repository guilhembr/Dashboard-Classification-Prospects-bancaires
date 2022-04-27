# coding=utf-8

import streamlit as st
import numpy as np
import pandas as pd
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report
from matplotlib.image import imread


def app():

    logo = imread("/Users/guilhemberthou/dev/P7_Scoring/app_pages/logo.png")

    st.sidebar.image(logo)
    st.sidebar.write("")
    st.sidebar.write("")
    
    st.title("Aperçu de la population de prospect")
        
    df = pd.read_csv("/Users/guilhemberthou/dev/P7_Scoring/dashboard_data/df_test.csv")
    
    #Sample data for Customer profile Analysis    
    colonnes_pandas_profiling = ["CODE_GENDER",
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
    st.write("Analyse exploratoire d'un échantillon du dataset de prospect (seules 11 des 101 variables du dataset sont présentées afin de comprendre le profil des prospects)")
    pr = df_pandas_profiling.profile_report()
    st_profile_report(pr)

if __name__ == "__main__":
    app()