CREATE PYTHON_UDF tpcx_ai_uc10_holistic(business_hour_norm REAL, amount_norm REAL) RETURNS INTEGER 
{"
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
def pyinitial():
    pass
def pyfun(*args):
    # Inference context setup
    model_file = '{HOME_PATH}/workload/tpcxai_datasets/sf10/models/uc10/uc10.python.model'
    uc10_model = joblib.load(model_file)
    # Data preprocess
    data = pd.DataFrame({
        'business_hour_norm': args[0],
        'amount_norm': args[1]
    })
    # Invoke model inference
    result = uc10_model.predict(data)
    return result
"};

CREATE PYTHON_UDF tpcx_ai_uc10_staged(business_hour_norm REAL, amount_norm REAL) RETURNS INTEGER 
{"
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Execute at the PF Initialization Stage
def pyinitial():
    # Inference context setup
    global uc10_model
    model_file = '{HOME_PATH}/workload/tpcxai_datasets/sf10/models/uc10/uc10.python.model'
    uc10_model = joblib.load(model_file)

# Execute at the PF Computation Stage
def pyfun(*args):
    # Data preprocess
    data = pd.DataFrame({
        'business_hour_norm': args[0],
        'amount_norm': args[1]
    })
    # Invoke model inference
    return uc10_model.predict(data)
"};