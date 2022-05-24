# Implement a Scoring Model (OC-P7)

## Overview

*Prêt à dépenser (eng. "Ready to Spend")*, is a financial company which offers consumer credits for people who have little or no loan history at all.

The company wishes to implement a **scoring credit tool** to calculate the likelihood that a customer reimburses its credit, and classifies the credit application granted or refused. It therefore wishes to develop a **classification algorithm** which would rely on various data sources (behavioral data, data from other financial institutions, etc.).

The dataset which will be used has been provided via a [kaggle competition](https://www.kaggle.com/c/home-credit-default-risk/data). 

In addition, customer relationship managers have brought up the fact that customers are increasingly asking for **transparency about the credit grant decision**. This demand for customer transparency is also very much in line with the values that the company wants to embody.

*Prêt à dépenser* therefore decides to develop an **interactive dashboard** so that customer relationship managers can be transparent about the credit decisions of the algorithm developed, as well as allowing their prospect to get access to their personal informations and compare it to the whole prospect population.

## Deliverables

### [Exploratory Notebook](https://github.com/guilhembr/P7_Scoring/blob/main/exploratory_notebook.ipynb) : 
- *1) Pre-processing* 
    - Cleaning of the dataset `application_train` as described in the [`Competition Notebook`](https://colab.research.google.com/drive/1uorVxsO816YOQMbkizlakGC21wS-xVHh#scrollTo=uN03kboJEVSi)
    - Exploratory Data Analysis (EDA)
- *2) Modeling*
    - Train/test/split (`app_test.csv` cannot be used as testing set because it does not contain the `TARGET` due to Kaggle competition context)
    - Training several classification models : DummyClassifier, Logistic Regression, Decision Tree, Random Forest, LGBM
    - Prediction performance analysis (Recall, F1-score, AUC, confusion matrix)
    -   Hyper-parameters tuning on dataset sample (Cross-Validation with GridSearchCV, `scoring`=`roc_auc`)
- *3) Managing the imbalance problem* 
    - Option 1 : Re-sampling (Under/Oversampling, Tomek links, SMOTE, SMOTETomek)
    - Option 2 : Hyper-parameter `class_weights` (Automatic tuning, manual tuning)
- *4) Selection of the best two models*
- *5) Feature engineering*
    - Feature creation
    - Automatic feature selection using `RFECV`(Recursive Feature Elimination with Cross-Validation)
- *6) Finding the best performance metric*
    - Creating a metric in line with business needs (`fbeta_score` optimization using `make_scorer`)
- *6) Decision threshold variation*
    - Choosing the optimal threshold to maximise metric result
- *7) Model interpretation*
    - Global interpretation (coefficients)
    - Local interpretation (SHAP : beeswarm, force plot) 
        - *please refer to Summary Notebook for most complete SHAP interpretations*

### [Summary Notebook](https://github.com/guilhembr/P7_Scoring/blob/main/model_training.ipynb) : *Cleaning, Pre-processing and modeling Notebook*
- *1) Pre-processing using functions*:
    - 1.1) Setting correct dtypes of features
    - 1.2) Creating new features based on business Knowledge
    - 1.3) Feature selection (based on Recursive Feature Elimination with Cross-Validation result from *Exploratory Notebook*)
    - 1.4) Encoding, Imputation and Standardization
        - Categorical variables treatment : OHE, `most frequent` imputation 
        - Numerical variables treatment : `median` imputation, `Standard` Scaling
- *2) Applying same pre-processing to `app_test` data*
- *3) Training `LogisticRegression` Model on pre-processed `app_train` data with hyperparameters optimized in* Exploratory Notebook
- *4) SHAP Interpretation graphs to be plotted in the Streamlit web interface*
- *4) Exporting objects to be used by the API and Streamlit web interface*
    - 4.1) `.csv` export of Pre-processed Datasets (5% sample of original datasets) (`dashboard_data` folder)
    - 4.2) `joblib` serialization of model and pre-processing transformers (`bin` folder)

### API Scoring prediction deployed on Heroku ([see the doc](https://projetoc-scoring.herokuapp.com/docs))
- API (back-end) used by Streamlit web interface (front-end) to get infos and scoring prediction of an unlabelled client selected in the dashboard (`api.py` ([see code](https://github.com/guilhembr/P7_Scoring/blob/main/api.py)))

### Dashboard Streamlit (*french*) ([access the frontend](https://share.streamlit.io/guilhembr/p7_scoring/main/app.py))
- Multi-pages streamlit dashboard code : `app.py` ([see code](https://github.com/guilhembr/P7_Scoring/blob/main/app.py)), `multiapp.py` and `app_pages folder`
- Overview of the training set sample with `Pandas Profiling` package
- Displaying unlabelled test client infos
- Request the API to obtain prediction of an unlabelled client selected by front-end user
![Client Jauge](https://github.com/guilhembr/P7_Scoring/blob/main/dashboard_screenshot/jauge_screenshot.png)
- Display prediction result and local interpretation (`SHAP Waterfall Plot`)
![SHAP Waterfall Client Prediction interpretation](https://github.com/guilhembr/P7_Scoring/blob/main/dashboard_screenshot/waterfall_screenshot.png)
- Compare client infos with training set population
![Population Comparison](https://github.com/guilhembr/P7_Scoring/blob/main/dashboard_screenshot/population_comparison_screenshot.png)

## Reference
Openclassrooms project from Data Scientist path.  

## Author
Guilhem Berthou (mentored by Pierre-Antoine Ganaye)

## License
Please see license.txt.
