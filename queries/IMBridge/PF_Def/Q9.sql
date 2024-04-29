CREATE PYTHON_UDF tpcx_ai_uc01_holistic(return_ratio REAL, frequency_ REAL) RETURNS INTEGER 
{"
import joblib
import numpy as np
import pandas as pd
# K-Means clustering
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
def pyinitial():
    pass
def pyfun(*args):
    # Inference context setup
    model_file = '{HOME_PATH}/workload/tpcxai_datasets/sf10/models/uc01/uc01.python.model'
    uc01_model = joblib.load(model_file)
    # Data preprocess
    feat = pd.DataFrame({
        'return_ratio': args[0],
        'frequency': args[1]
    })
    # Invoke model inference
    return uc01_model.predict(feat)
"};

CREATE PYTHON_UDF tpcx_ai_uc01_staged(return_ratio REAL, frequency_ REAL) RETURNS INTEGER 
{"
import joblib
import numpy as np
import pandas as pd
# K-Means clustering
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
# Execute at the PF Initialization Stage
def pyinitial():
    # Inference context setup
    global uc01_model
    model_file = '{HOME_PATH}/workload/tpcxai_datasets/sf10/models/uc01/uc01.python.model'
    uc01_model = joblib.load(model_file)
# Execute at the PF Computation Stage
def pyfun(*args):
    # Data preprocess
    feat = pd.DataFrame({
        'return_ratio': args[0],
        'frequency': args[1]
    })
    # Invoke model inference
    return uc01_model.predict(feat)
"};

