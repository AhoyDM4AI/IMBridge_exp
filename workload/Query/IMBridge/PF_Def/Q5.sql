CREATE PYTHON_UDF hospital_sklearn_lr_holistic(hematocrit REAL, neutrophils REAL, sodium REAL, glucose REAL, bloodureanitro REAL, creatinine REAL, bmi REAL, pulse INTEGER, respiration REAL, secondarydiagnosisnonicd9 INTEGER, rcount STRING, gender STRING, dialysisrenalendstage INTEGER, asthma INTEGER, irondef INTEGER, pneum INTEGER, substancedependence INTEGER, psychologicaldisordermajor INTEGER, depress INTEGER, psychother INTEGER, fibrosisandother INTEGER, malnutrition INTEGER, hemo INTEGER) RETURNS INTEGER 
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
    scaler_path = '{HOME_PATH}/workload/raven_datasets/Hospital/hospital_standard_scale_model.pkl'
    enc_path = '{HOME_PATH}/workload/raven_datasets/Hospital/hospital_one_hot_encoder.pkl'
    model_path = '{HOME_PATH}/workload/raven_datasets/Hospital/hospital_lr_model.pkl'
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    with open(enc_path, 'rb') as f:
        enc = pickle.load(f)
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    # Data preprocess
    data = np.column_stack(args)
    data = np.split(data, np.array([10]), axis = 1)
    numerical = data[0]
    categorical = data[1]
    X = np.hstack((scaler.transform(numerical), enc.transform(categorical).toarray()))
    # Invoke model inference
    return model.predict(X)
"};

CREATE PYTHON_UDF hospital_sklearn_lr_staged(hematocrit REAL, neutrophils REAL, sodium REAL, glucose REAL, bloodureanitro REAL, creatinine REAL, bmi REAL, pulse INTEGER, respiration REAL, secondarydiagnosisnonicd9 INTEGER, rcount STRING, gender STRING, dialysisrenalendstage INTEGER, asthma INTEGER, irondef INTEGER, pneum INTEGER, substancedependence INTEGER, psychologicaldisordermajor INTEGER, depress INTEGER, psychother INTEGER, fibrosisandother INTEGER, malnutrition INTEGER, hemo INTEGER) RETURNS INTEGER 
{"
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
# Execute at the PF Initialization Stage
def pyinitial():
    # Inference context setup
    global hospital_scaler, hospital_enc, hospital_model
    scaler_path = '{HOME_PATH}/workload/raven_datasets/Hospital/hospital_standard_scale_model.pkl'
    enc_path = '{HOME_PATH}/workload/raven_datasets/Hospital/hospital_one_hot_encoder.pkl'
    model_path = '{HOME_PATH}/workload/raven_datasets/Hospital/hospital_lr_model.pkl'
    with open(scaler_path, 'rb') as f:
        hospital_scaler = pickle.load(f)
    with open(enc_path, 'rb') as f:
        hospital_enc = pickle.load(f)
    with open(model_path, 'rb') as f:
        hospital_model = pickle.load(f)
# Execute at the PF Computation Stage
def pyfun(*args):
    # Data preprocess
    data = np.column_stack(args)
    data = np.split(data, np.array([10]), axis = 1)
    numerical = data[0]
    categorical = data[1]
    X = np.hstack((hospital_scaler.transform(numerical), hospital_enc.transform(categorical).toarray()))
    # Invoke model inference
    return hospital_model.predict(X)
"};