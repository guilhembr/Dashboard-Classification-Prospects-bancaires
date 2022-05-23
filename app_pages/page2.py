import streamlit as st
import numpy as np
import pandas as pd
from matplotlib.image import imread
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns
import requests
import joblib
import shap
# import streamlit.components.v1 as components

shap.initjs()
st.set_page_config(layout="wide")
st.set_option('deprecation.showPyplotGlobalUse', False)

########################################################
# Session for the API
########################################################
@st.experimental_singleton(suppress_st_warning=True)
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
@st.experimental_memo(suppress_st_warning=True)
def client():
	#Getting Client details
	response = fetch(session, f"http://projetoc-scoring.herokuapp.com/api/clients")
	if response:
		return response["clientsId"]
	else:
		return "Error"

@st.experimental_memo(suppress_st_warning=True)
def client_details(id):
	#Getting Client details
	response = fetch(session,f"http://projetoc-scoring.herokuapp.com/api/clients/{id}")
	if response:
		return response
	else:
		return "Error"

@st.experimental_memo(suppress_st_warning=True)
def client_prediction(id):
    #Getting Client prediction
	response = fetch(session, f"http://projetoc-scoring.herokuapp.com/api/clients/{id}/prediction")
	if response:
		return response
	else:
		return "Error"

########################################################
# Functions to automate the graphs
########################################################

@st.experimental_singleton(suppress_st_warning=True)
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

@st.experimental_singleton(suppress_st_warning=True)
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

@st.experimental_singleton(suppress_st_warning=True)
def display_charts(df,client):
	"""Affichage des graphes de comparaison pour le client sélectionné """
	row1_1,row1_2,row1_3 = st.columns(3)
	st.write('')
	row2_10,row2_2,row2_3 = st.columns(3)
	st.write('')
	row3_1, row3_2, row3_3 = st.columns(3)
	
	chart_bar("Niveau d'études",row1_1, df,'NAME_EDUCATION_TYPE',client)
	chart_kde("Ratio Revenu/Annuité",row1_2, df,'annuity_income_ratio',client)
	chart_kde("Revenus totaux",row1_3, df,'AMT_INCOME_TOTAL',client)
	
	chart_kde("Apport",row2_10, df,'credit_downpayment',client)
	chart_kde("Durée d'activité pro.",row2_2, df,'DAYS_EMPLOYED',client)
	chart_bar("Sexe",row2_3,df,'CODE_GENDER',client)

	chart_bar("Propriétaire d'un véhicule",row3_1,df,'FLAG_OWN_CAR',client)
	chart_bar("Répartition du statut professionel",row3_2,df,'NAME_INCOME_TYPE',client)
	chart_bar("Répartition du type de logement",row3_3,df,'NAME_HOUSING_TYPE',client)

@st.experimental_singleton(suppress_st_warning=True)
def color(pred):
	'''Définition de la couleur selon la prédiction'''
	if pred=='Approved':
		col='Green'
	else :
		col='Red'
	return col

# def st_shap(plot, height=None):
# 	"""Fonction permettant l'affichage de graphique shap values"""
# 	shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
# 	components.html(shap_html, height=height)
 
########################################################
#Main function : app()
######################################################## 

def app():
	"""Fonction générant la page 2 du dashboard. Ne prend pas de paramètre en entrée.
	"""
	logo = imread("./app_pages/logo.png")

	st.sidebar.image(logo)
	st.sidebar.write("")
	st.sidebar.write("")

#Client Infos
#-------------------------------------------------------

	#Get Client
	client_id = st.sidebar.selectbox("Client Id List", client())
	
	#Get Prediction for selected client
	prediction = client_prediction(client_id)
	prediction = pd.read_json(prediction)
	
	#calculate decision and convert array to string
	threshold = 0.48
	decision = str(np.where(prediction["pred"].iloc[0]>threshold,"Rejected","Approved"))
	
	#changing sign of features
	prediction["credit_downpayment"] = abs(prediction["credit_downpayment"])
	prediction["DAYS_EMPLOYED"] = abs(prediction["DAYS_EMPLOYED"])	
	
	#Infos Client
	st.sidebar.markdown("ID client: " + str(client_id))
	
	st.sidebar.markdown("Sexe: " + prediction["CODE_GENDER"].iloc[0])
	st.sidebar.markdown("Statut familial: " + prediction["NAME_FAMILY_STATUS"].iloc[0])
	st.sidebar.markdown("Enfants: " + str(prediction["CNT_CHILDREN"].iloc[0].astype("int64")))
	st.sidebar.markdown("Age: " + str(prediction["AGE_INT"].iloc[0].astype("int64")))
	st.sidebar.markdown("Statut pro.: " + prediction["NAME_INCOME_TYPE"].iloc[0])
	st.sidebar.markdown("Durée d'activité: " + str(prediction["DAYS_EMPLOYED"].iloc[0]))
	st.sidebar.markdown("Type de logement: " + prediction["NAME_HOUSING_TYPE"].iloc[0])
	st.sidebar.markdown("Véhicule: " + prediction["FLAG_OWN_CAR"].iloc[0])
	st.sidebar.markdown("Revenus: " + str(prediction["AMT_INCOME_TOTAL"].iloc[0]))
	st.sidebar.markdown("Ratio Revenus/Annuité: " + str(round(prediction["annuity_income_ratio"].iloc[0],2)))
	st.sidebar.markdown("Apport: " + str(prediction["credit_downpayment"].iloc[0]))
	
#Client Score 
#-------------------------------------------------------

	st.subheader("Score Client")
	st.write("Prédiction du score de solvabilité du client. Le seuil d'approbation est fixé à 0,48.")
	
	fig = go.Figure(go.Indicator(
	mode = "gauge+number+delta",
    value = prediction["pred"].iloc[0],
    number = {'font':{'size':48}},
    domain = {'x': [0, 1], 'y': [0, 1]},
    title = {'text': decision, 'font': {'size': 28, 'color':color(decision)}},
    delta = {'reference': 0.48, 'increasing': {'color': "red"},'decreasing':{'color':'green'}},
    gauge = {
        'axis': {'range': [0,1], 'tickcolor': color(decision)},
        'bar': {'color': color(decision)},
        'steps': [
            {'range': [0,0.48], 'color': 'lightgreen'},
            {'range': [0.48,1], 'color': 'lightcoral'}],
        'threshold': {
            'line': {'color': "black", 'width': 5},
            'thickness': 1,
            'value': 0.48}}))

	fig.update_layout(height=500, width=1200)

	st.plotly_chart(fig, height=500, width=1200)
 
#SHAP Client
#-------------------------------------------------------

	df = pd.read_csv("./dashboard_data/df_train.csv")
	df_test = pd.read_csv("./dashboard_data/df_test.csv")
	df_test_cat_features = pd.read_csv("./dashboard_data/df_test_cat_features.csv")
	df_test_num_features = pd.read_csv("./dashboard_data/df_test_num_features.csv")

	ohe = joblib.load("./bin/ohe.joblib")
	categorical_imputer = joblib.load("./bin/categorical_imputer.joblib")
	simple_imputer = joblib.load("./bin/simple_imputer.joblib")
	scaler = joblib.load("./bin/scaler.joblib")
	model = joblib.load("./bin/model.joblib")
	
	#---------------------------------------------------------------------
	#data pre-processing (training set)

	list_cat_features = df_test_cat_features.columns.to_list()
	list_num_features = df_test_num_features.columns.to_list()
 
	#SimpleImputing (most frequent) and ohe of categorical features
	cat_array = categorical_imputer.transform(df[list_cat_features])
	cat_array = ohe.transform(cat_array).todense()

	#SimpleImputing (median) and StandardScaling of numerical features
	num_array = simple_imputer.transform(df[list_num_features])
	num_array = scaler.transform(num_array)

	#concatenate
	X_train = np.concatenate([cat_array, num_array], axis=1)
	X_train = np.asarray(X_train)

	#building dataframe with post-preprocessed data (training set)
	cat_features_list_after_ohe = ohe.get_feature_names_out(list_cat_features).tolist()
	features_list_after_prepr = cat_features_list_after_ohe + list_num_features
	ohe_dataframe = pd.DataFrame(X_train, columns=features_list_after_prepr)

	#---------------------------------------------------------------------
	#data pre-processing (test set)

	#SimpleImputing (most frequent) and ohe of categorical features
	cat_array = categorical_imputer.transform(df_test[list_cat_features])
	cat_array = ohe.transform(cat_array).todense()

	#SimpleImputing (median) and StandardScaling of numerical features
	num_array = simple_imputer.transform(df_test[list_num_features])
	num_array = scaler.transform(num_array)

	#concatenate
	X = np.concatenate([cat_array, num_array], axis=1)
	X = np.asarray(X)

	#building dataframe with post-preprocessed data (testing set)
	cat_features_list_after_ohe = ohe.get_feature_names_out(list_cat_features).tolist()
	features_list_after_prepr_test = cat_features_list_after_ohe + list_num_features
	ohe_dataframe_test = pd.DataFrame(X, columns=features_list_after_prepr_test)

	#---------------------------------------------------------------------
	#shap values
	sub_sampled_train_data = shap.sample(ohe_dataframe, 50)
	log_reg_explainer = shap.KernelExplainer(model.predict_proba, sub_sampled_train_data)

	#shap values
	ohe_dataframe_test["SK_ID_CURR"] = df_test["SK_ID_CURR"]
	sub_sampled_test_data = ohe_dataframe_test[ohe_dataframe_test["SK_ID_CURR"] == client_id].drop(columns="SK_ID_CURR")
	sub_sampled_test_data = sub_sampled_test_data.values.reshape(1,-1)
	shap_vals = log_reg_explainer.shap_values(sub_sampled_test_data)

	#Layout
	st.subheader("Explication du score client")
 
	st.write("Analyse des principales variables ayant contribuées à la prédiction réalisée par le modèle.")

	st.pyplot(shap.plots._waterfall.waterfall_legacy(log_reg_explainer.expected_value[1],
										shap_vals[1][0],
										sub_sampled_test_data[0],
										feature_names=features_list_after_prepr_test,
										max_display=10
          ))
 

#Comparaison with Training population
#-------------------------------------------------------

	#changing type of Data comparison features
	df["CNT_CHILDREN"] = df["CNT_CHILDREN"].astype("int64")
	df["AGE_INT"] = df["AGE_INT"].astype("int64")
	
	#changing sign of features
	df["credit_downpayment"] = abs(df["credit_downpayment"])
	
	df['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)
	df["DAYS_EMPLOYED"] = abs(df["DAYS_EMPLOYED"])	
	
	#renaming pred column in prediction row to TARGET to match with training set
	prediction.rename(columns={"pred": "TARGET"})
	
	#concatenate training set and prediction row
	frames = [prediction, df]
	df = pd.concat(frames)
	
	#Reset index
	df = df.reset_index(drop=True)

	#Layout
	st.subheader("Comparaison clientèle")
 
	st.write("Analyse des principales caractéristiques du prospect par rapport à la population.")
	st.markdown("""
			 La variable TARGET indique l'historique de remboursement:  
			 - 0 = solvable 
			 - 1 = insolvable """
			 )

	#display charts
	idx_client = df.index[df["SK_ID_CURR"]==client_id][0]
	display_charts(df, idx_client)
 
if __name__ == "__main__":
	app()