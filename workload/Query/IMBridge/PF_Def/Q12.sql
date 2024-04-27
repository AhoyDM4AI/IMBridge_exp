CREATE PYTHON_UDF tpcx_ai_uc06_holistic(smart_5_raw REAL, smart_10_raw REAL, smart_184_raw REAL, smart_187_raw REAL, smart_188_raw REAL, smart_197_raw REAL, smart_198_raw REAL) RETURNS INTEGER 
{"
import joblib
import numpy as np
import pandas as pd
from duckdb.typing import BIGINT, DOUBLE, FLOAT, VARCHAR
from imblearn.over_sampling import ADASYN
from imblearn.pipeline import Pipeline as Imb_Pipeline
import joblib
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
def pyinitial():
    pass
def pyfun(*args):
    # Inference context setup
    model_file = '{HOME_PATH}/workload/tpcxai_datasets/sf10/models/uc06/uc06.python.model'
    uc06_model = joblib.load(model_file)
    # Data preprocess
    data = pd.DataFrame({
        'smart_5_raw': args[0],
        'smart_10_raw': args[1],
        'smart_184_raw': args[2],
        'smart_187_raw': args[3],
        'smart_188_raw': args[4],
        'smart_197_raw': args[5],
        'smart_198_raw': args[6]
    })
    # Invoke model inference
    return uc06_model.predict(data)
"};

CREATE PYTHON_UDF tpcx_ai_uc06_staged(smart_5_raw REAL, smart_10_raw REAL, smart_184_raw REAL, smart_187_raw REAL, smart_188_raw REAL, smart_197_raw REAL, smart_198_raw REAL) RETURNS INTEGER 
{"
import joblib
import numpy as np
import pandas as pd
from duckdb.typing import BIGINT, DOUBLE, FLOAT, VARCHAR
from imblearn.over_sampling import ADASYN
from imblearn.pipeline import Pipeline as Imb_Pipeline
import joblib
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
# Execute at the PF Initialization Stage
def pyinitial():
    # Inference context setup
    global uc06_model
    model_file = '{HOME_PATH}/workload/tpcxai_datasets/sf10/models/uc06/uc06.python.model'
    uc06_model = joblib.load(model_file)
# Execute at the PF Computation Stage
def pyfun(*args):
    # Data preprocess
    data = pd.DataFrame({
        'smart_5_raw': args[0],
        'smart_10_raw': args[1],
        'smart_184_raw': args[2],
        'smart_187_raw': args[3],
        'smart_188_raw': args[4],
        'smart_197_raw': args[5],
        'smart_198_raw': args[6]
    })
    # Invoke model inference
    return uc06_model.predict(data)
"};