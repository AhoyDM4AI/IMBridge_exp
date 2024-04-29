CREATE PYTHON_UDF flights_sklearn_rf_holistic(slatitude REAL, slongitude REAL, dlatitude REAL, dlongitude REAL, name1 INTEGER, name2 STRING, name4 STRING, acountry STRING, active_ STRING, scity STRING, scountry STRING, stimezone INTEGER, sdst STRING, dcity STRING, dcountry STRING, dtimezone INTEGER, ddst STRING) RETURNS INTEGER 
{"
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
def pyinitial():
    pass
def pyfun(*args):
    # Inference context setup
    scaler_path = '{HOME_PATH}/workload/raven_datasets/Flights/flights_standard_scale_model.pkl'
    enc_path = '{HOME_PATH}/workload/raven_datasets/Flights/flights_one_hot_encoder.pkl'
    model_path = '{HOME_PATH}/workload/raven_datasets/Flights/flights_rf_model.pkl'
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    with open(enc_path, 'rb') as f:
        enc = pickle.load(f)
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    # Data preprocess
    data = np.column_stack(args)
    data = np.split(data, np.array([4]), axis = 1)
    numerical = data[0]
    categorical = data[1]
    X = np.hstack((scaler.transform(numerical), enc.transform(categorical).toarray()))
    # Invoke model inference
    return model.predict(X)
"};

CREATE PYTHON_UDF flights_sklearn_rf_staged(slatitude REAL, slongitude REAL, dlatitude REAL, dlongitude REAL, name1 INTEGER, name2 STRING, name4 STRING, acountry STRING, active_ STRING, scity STRING, scountry STRING, stimezone INTEGER, sdst STRING, dcity STRING, dcountry STRING, dtimezone INTEGER, ddst STRING) RETURNS INTEGER 
{"
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
# Execute at the PF Initialization Stage
def pyinitial():
    # Inference context setup
    global flights_scaler, flights_enc, flights_model
    scaler_path = '{HOME_PATH}/workload/raven_datasets/Flights/flights_standard_scale_model.pkl'
    enc_path = '{HOME_PATH}/workload/raven_datasets/Flights/flights_one_hot_encoder.pkl'
    model_path = '{HOME_PATH}/workload/raven_datasets/Flights/flights_rf_model.pkl'
    with open(scaler_path, 'rb') as f:
        flights_scaler = pickle.load(f)
    with open(enc_path, 'rb') as f:
        flights_enc = pickle.load(f)
    with open(model_path, 'rb') as f:
        flights_model = pickle.load(f)
# Execute at the PF Computation Stage
def pyfun(*args):
    # Data preprocess
    data = np.column_stack(args)
    data = np.split(data, np.array([4]), axis = 1)
    numerical = data[0]
    categorical = data[1]
    X = np.hstack((flights_scaler.transform(numerical), flights_enc.transform(categorical).toarray()))
    # Invoke model inference
    return flights_model.predict(X)
"};