CREATE PYTHON_UDF tpcx_ai_uc07_holistic(user_id INTEGER, item_id INTEGER) RETURNS REAL 
{"
import joblib
import numpy as np
import pandas as pd
from duckdb.typing import BIGINT, DOUBLE, FLOAT, VARCHAR
import joblib
from surprise import SVD
from surprise import Dataset
from surprise.reader import Reader

def pyinitial():
    pass

def pyfun(*args):
    # Inference context setup
    model_file = '{HOME_PATH}/workload/tpcxai_datasets/sf10/models/uc07/uc07.python.model'
    uc07_model = joblib.load(model_file)
    # Data preprocess
    user_id = args[0]
    item_id = args[1]
    ratings = []
    for i in range(len(user_id)):
        # Invoke model inference
        rating = uc07_model.predict(user_id[i], item_id[i]).est
        ratings.append(rating)
    return np.array(ratings)
"};

CREATE PYTHON_UDF tpcx_ai_uc07_staged(user_id INTEGER, item_id INTEGER) RETURNS REAL 
{"
import joblib
import numpy as np
import pandas as pd
from duckdb.typing import BIGINT, DOUBLE, FLOAT, VARCHAR
import joblib
from surprise import SVD
from surprise import Dataset
from surprise.reader import Reader

# Execute at the PF Initialization Stage
def pyinitial():
    # Inference context setup
    global uc07_model
    model_file = '{HOME_PATH}/workload/tpcxai_datasets/sf10/models/uc07/uc07.python.model'
    uc07_model = joblib.load(model_file)

# Execute at the PF Computation Stage
def pyfun(*args):
    # Data preprocess
    user_id = args[0]
    item_id = args[1]
    ratings = []
    for i in range(len(user_id)):
        # Invoke model inference
        rating = uc07_model.predict(user_id[i], item_id[i]).est
        ratings.append(rating)
    return np.array(ratings)
"};