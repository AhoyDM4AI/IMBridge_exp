CREATE PYTHON_UDF creditcard_lightgbm_gb_holistic(V1 REAL, V2 REAL, V3 REAL, V4 REAL, V5 REAL, V6 REAL, V7 REAL, V8 REAL, V9 REAL, V10 REAL, V11 REAL, V12 REAL, V13 REAL, V14 REAL, V15 REAL, V16 REAL, V17 REAL, V18 REAL, V19 REAL, V20 REAL, V21 REAL, V22 REAL, V23 REAL, V24 REAL, V25 REAL, V26 REAL, V27 REAL, V28 REAL, Amount REAL) RETURNS INTEGER 
{"
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
def pyinitial():
    pass
def pyfun(*args):
    # Inference context setup
    scaler_path = '{HOME_PATH}/workload/raven_datasets/Credit_Card/creditcard_standard_scale_model.pkl'
    model_path = '{HOME_PATH}/workload/raven_datasets/Credit_Card/creditcard_lgb_model.txt'
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    model = lgb.Booster(model_file=model_path)
    # Data preprocess
    numerical = np.column_stack(args)
    X = scaler.transform(numerical)
    # Invoke model inference
    return model.predict(X)
"};

CREATE PYTHON_UDF creditcard_lightgbm_gb_staged(V1 REAL, V2 REAL, V3 REAL, V4 REAL, V5 REAL, V6 REAL, V7 REAL, V8 REAL, V9 REAL, V10 REAL, V11 REAL, V12 REAL, V13 REAL, V14 REAL, V15 REAL, V16 REAL, V17 REAL, V18 REAL, V19 REAL, V20 REAL, V21 REAL, V22 REAL, V23 REAL, V24 REAL, V25 REAL, V26 REAL, V27 REAL, V28 REAL, Amount REAL) RETURNS INTEGER 
{"
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
# Execute at the PF Initialization Stage
def pyinitial():
    # Inference context setup
    global creditcard_scaler, creditcard_model
    scaler_path = '{HOME_PATH}/workload/raven_datasets/Credit_Card/creditcard_standard_scale_model.pkl'
    model_path = '{HOME_PATH}/workload/raven_datasets/Credit_Card/creditcard_lgb_model.txt'
    with open(scaler_path, 'rb') as f:
        creditcard_scaler = pickle.load(f)
    creditcard_model = lgb.Booster(model_file=model_path)
# Execute at the PF Computation Stage
def pyfun(*args):
    # Data preprocess
    numerical = np.column_stack(args)
    X = creditcard_scaler.transform(numerical)
    # Invoke model inference
    return creditcard_model.predict(X)
"};