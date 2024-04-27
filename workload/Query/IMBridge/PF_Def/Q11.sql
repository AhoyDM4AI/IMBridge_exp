CREATE PYTHON_UDF tpcx_ai_uc04_holistic(text_ STRING) RETURNS INTEGER 
{"
import joblib
import numpy as np
import pandas as pd
from duckdb.typing import BIGINT, DOUBLE, FLOAT, VARCHAR
from sklearn import feature_extraction, naive_bayes
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
def pyinitial():
    pass
def pyfun(*args):
    # Inference context setup
    model_file = '{HOME_PATH}/workload/tpcxai_datasets/sf10/models/uc04/uc04.python.model'
    uc04_model = joblib.load(model_file)
    # Data preprocess
    data = pd.DataFrame({
        'text': args[0]
    })
    # Invoke model inference
    return uc04_model.predict(data['text'])
"};

CREATE PYTHON_UDF tpcx_ai_uc04_staged(text_ STRING) RETURNS INTEGER 
{"
import joblib
import numpy as np
import pandas as pd
from duckdb.typing import BIGINT, DOUBLE, FLOAT, VARCHAR
from sklearn import feature_extraction, naive_bayes
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
# Execute at the PF Initialization Stage
def pyinitial():
    # Inference context setup
    global uc04_model
    model_file = '{HOME_PATH}/workload/tpcxai_datasets/sf10/models/uc04/uc04.python.model'
    uc04_model = joblib.load(model_file)
# Execute at the PF Computation Stage
def pyfun(*args):
    # Data preprocess
    data = pd.DataFrame({
        'text': args[0]
    })
    # Invoke model inference
    return uc04_model.predict(data['text'])
"};