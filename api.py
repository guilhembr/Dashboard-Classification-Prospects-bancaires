import uvicorn ##ASGI
from fastapi import FastAPI, HTTPException
import json
# from pydantic import BaseModel, create_model
import joblib
import numpy as np
import pandas as pd


#######################################################################
# Loading data (to predict) and model
#---------------------------------------------------------------------

#loading data
df_test = pd.read_csv("./dashboard_data/df_test.csv")
df_test_cat_features = pd.read_csv("./dashboard_data/df_test_cat_features.csv")
df_test_num_features = pd.read_csv("./dashboard_data/df_test_num_features.csv")

#load serialized objects
# data_dict = joblib.load("./bin/data_dict.joblib")
ohe = joblib.load("./bin/ohe.joblib")
categorical_imputer = joblib.load("./bin/categorical_imputer.joblib")
simple_imputer = joblib.load("./bin/simple_imputer.joblib")
scaler = joblib.load("./bin/scaler.joblib")
model = joblib.load("./bin/model.joblib")

#---------------------------------------------------------------------
#data pre-processing (test set)

#imputation
cat_features = categorical_imputer.transform(df_test_cat_features)
num_features = simple_imputer.transform(df_test_num_features)

#One hot encoding categorical variables
cat_array = ohe.transform(cat_features).todense()
cat_array = np.asarray(cat_array)

#Standard Scaling numerical variables
num_array = scaler.transform(num_features)

#concatenate
X_test = np.concatenate([cat_array, num_array], axis=1)
X_test = np.asarray(X_test)

#######################################################################
#create fast API instance
app = FastAPI()

@app.get("/api/clients")
async def clients_id():
    """Endpoint to get all clients id

    Returns:
        list: clients_id
    """
    clients_id = df_test["SK_ID_CURR"].to_list()
    
    return {"clientsId": clients_id}  

@app.get("/api/clients/{id}")
async def client_details(id: int):
    """Endpoint to get client's details

    Args:
        id (int): client id in the test set

    Returns:
        client (dict): client's details
    """
    clients_id = df_test["SK_ID_CURR"].to_list()
    
    if id not in clients_id:
        raise HTTPException(status_code=404, detail="client's id not found")
    else:
        #filtering by client's id
        idx = df_test[df_test["SK_ID_CURR"] == id].index[0]
          
        client = {
            "clientId" : str(df_test.loc[idx,"SK_ID_CURR"]),
            "sexe": str(df_test.loc[idx,"CODE_GENDER"]),
            "statut familial": str(df_test.loc[idx,"NAME_FAMILY_STATUS"]),
            "enfants": str(df_test.loc[idx,"CNT_CHILDREN"]),
            "age": str(df_test.loc[idx,"AGE_INT"]),
            "statut pro": str(df_test.loc[idx,"NAME_INCOME_TYPE"]),
            "niveau d'études": str(df_test.loc[idx,"NAME_EDUCATION_TYPE"])
        }
        
        return client

@app.get("/api/clients/{id}/prediction")
async def predict(id: int):
    """Generate prediction and SHAP inputs for a selected client

    Args:
        id (int): SK_ID_CURR selected

    Raises:
        HTTPException: If SK_ID_CURR not found

    Returns:
        prediction_by_id (json) : dataset including prediction for the selected client
        log_reg_explainer (explainer) : SHAP Kernel Explainer
        shap_vals (list) : List of 2 arrays for 0 and 1 prediction classes
        features_list_after_prepr_test (list) : list of feature names
    """
    
    clients_id = df_test["SK_ID_CURR"].to_list()
    
    if id not in clients_id:
        raise HTTPException(status_code=404, detail="client's id not found")
    else:
       
        #prediction
        result_proba = model.predict_proba(X_test)
        y_pred_proba = result_proba[:, 1]

        
        df_test["pred"] = y_pred_proba
        df_test["pred"] = round(df_test["pred"].astype(np.float64),4)
        
        #filtering by client's id
        df_test_by_id = df_test[df_test["SK_ID_CURR"] == id]
        
        prediction_by_id = df_test_by_id.to_json()
        
        return prediction_by_id

#------------Cheat Sheet-----------------
    
#Run the API with uvicorn
#uvicorn api:app --reload  
    
#requirements.txt
#pip list --format=freeze > requirements.txt

#kill processes on port : kill -9 $(lsof -t -i:"8000")

#heroku login
# #heroku git:remote -a projetoc-scoring
#git push heroku main
#heroku ps:scale web=1

#------------FUTURE IMPROVEMENTS-----------------

# #create model
# ScoringModel = create_model(
#     "ScoringModel",
#     **data_dict,
#     __base__=BaseModel,
# )

# ScoringModel.update_forward_refs()

# @app.post("/scoring")
# async def predict_scoring(item: ScoringModel):
#     item_dict = item.dict()

#     df = pd.DataFrame(data=[item_dict.values()], columns=item_dict.keys())
#     df = df.astype(object)

#     return json.dumps(model.predict_proba(X).tolist())


