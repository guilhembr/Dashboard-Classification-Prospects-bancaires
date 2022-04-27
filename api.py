# 1. import modules
import uvicorn ##ASGI
from json.tool import main
from fastapi import FastAPI
from pydantic import BaseModel, create_model
import joblib
import numpy as np
import pandas as pd

# 2. create fast API instance
app = FastAPI()

# 3. load data
data_dict = joblib.load("./bin/data_dict.joblib")
ohe = joblib.load("./bin/ohe.joblib")
imp_mean = joblib.load("./bin/imp_mean.joblib")
scaler = joblib.load("./bin/scaler.joblib")
model = joblib.load("./bin/model.joblib")

ScoringModel = create_model(
    "ScoringModel",
    **data_dict,
    __base__=BaseModel,
)

ScoringModel.update_forward_refs()

list_cat_features = ["NAME_CONTRACT_TYPE",
    "CODE_GENDER",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
    "NAME_TYPE_SUITE",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "REGION_POPULATION_RELATIVE",
    "FLAG_MOBIL",
    "FLAG_EMP_PHONE",
    "FLAG_WORK_PHONE",
    "FLAG_CONT_MOBILE",
    "FLAG_PHONE",
    "FLAG_EMAIL",
    "OCCUPATION_TYPE",
    "CNT_FAM_MEMBERS",
    "REGION_RATING_CLIENT",
    "WEEKDAY_APPR_PROCESS_START",
    "HOUR_APPR_PROCESS_START",
    "REG_REGION_NOT_LIVE_REGION",
    "REG_REGION_NOT_WORK_REGION",
    "LIVE_REGION_NOT_WORK_REGION",
    "REG_CITY_NOT_LIVE_CITY",
    "REG_CITY_NOT_WORK_CITY",
    "LIVE_CITY_NOT_WORK_CITY",
    "ORGANIZATION_TYPE",
    "FONDKAPREMONT_MODE",
    "HOUSETYPE_MODE",
    "WALLSMATERIAL_MODE",
    "EMERGENCYSTATE_MODE",
    "FLAG_DOCUMENT_2",
    "FLAG_DOCUMENT_3",
    "FLAG_DOCUMENT_4",
    "FLAG_DOCUMENT_5",
    "FLAG_DOCUMENT_6",
    "FLAG_DOCUMENT_7",
    "FLAG_DOCUMENT_8",
    "FLAG_DOCUMENT_9",
    "FLAG_DOCUMENT_10",
    "FLAG_DOCUMENT_11",
    "FLAG_DOCUMENT_12",
    "FLAG_DOCUMENT_13",
    "FLAG_DOCUMENT_14",
    "FLAG_DOCUMENT_15",
    "FLAG_DOCUMENT_16",
    "FLAG_DOCUMENT_17",
    "FLAG_DOCUMENT_18",
    "FLAG_DOCUMENT_19",
    "FLAG_DOCUMENT_20",
    "FLAG_DOCUMENT_21"
]

list_num_features = [
    "CNT_CHILDREN",
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "DAYS_EMPLOYED",
    "DAYS_REGISTRATION",
    "DAYS_ID_PUBLISH",
    "OWN_CAR_AGE",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "APARTMENTS_AVG",
    "BASEMENTAREA_AVG",
    "YEARS_BEGINEXPLUATATION_AVG",
    "YEARS_BUILD_AVG",
    "COMMONAREA_AVG",
    "ELEVATORS_AVG",
    "ENTRANCES_AVG",
    "FLOORSMAX_AVG",
    "FLOORSMIN_AVG",
    "LANDAREA_AVG",
    "LIVINGAPARTMENTS_AVG",
    "LIVINGAREA_AVG",
    "NONLIVINGAPARTMENTS_AVG",
    "NONLIVINGAREA_AVG",
    "APARTMENTS_MODE",
    "YEARS_BEGINEXPLUATATION_MODE",
    "FLOORSMIN_MODE",
    "LIVINGAREA_MODE",
    "LANDAREA_MEDI",
    "TOTALAREA_MODE",
    "OBS_30_CNT_SOCIAL_CIRCLE",
    "DEF_30_CNT_SOCIAL_CIRCLE",
    "DEF_60_CNT_SOCIAL_CIRCLE",
    "DAYS_LAST_PHONE_CHANGE",
    "AMT_REQ_CREDIT_BUREAU_HOUR",
    "AMT_REQ_CREDIT_BUREAU_DAY",
    "AMT_REQ_CREDIT_BUREAU_WEEK",
    "AMT_REQ_CREDIT_BUREAU_MON",
    "AMT_REQ_CREDIT_BUREAU_QRT",
    "AMT_REQ_CREDIT_BUREAU_YEAR",
    "DAYS_EMPLOYED_ANOM",
    "AGE_INT",
    "annuity_income_ratio",
    "credit_annuity_ratio",
    "credit_goods_price_ratio",
    "credit_downpayment"
]


@app.get("/")
def index():
    return {"message": "Hello Guilhem"}  

# 5. Route with a single parameter, returns the parameter within a message located at /Scoring
@app.post("/scoring")
async def predict_scoring(model: ScoringModel):
    item_dict = model.dict()

    df = pd.DataFrame(data=[item_dict.values()], columns=item_dict.keys())
    df = df.astype(object)

    cat_array = df[list_cat_features]
    cat_array = ohe.transform(cat_array).todense()

    num_array = df[list_num_features].to_numpy()
    num_array = scaler.transform(num_array)

    X = np.concatenate[cat_array, num_array]
    X = np.asarray(X)

    return json.dumps(model.predict_proba(X).tolist())

# 6. Run the API with uvicorn
#uvicorn api:app --reload  

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1")