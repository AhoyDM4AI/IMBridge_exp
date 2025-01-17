import pickle

import duckdb
import numpy as np
from duckdb.typing import BIGINT, DOUBLE, FLOAT, VARCHAR
import pandas as pd
import time
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

name = "q3"

con = duckdb.connect("imbridge2.db")

scaler_path = '/home/test_raven/Credit_Card/creditcard_standard_scale_model.pkl'
model_path = '/home/test_raven/Credit_Card/creditcard_xgb_model.json'
with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)
model = xgb.Booster(model_file=model_path)


def udf(V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11, V12, V13, V14, V15, V16, V17, V18, V19, V20, V21, V22, V23, V24, V25, V26, V27, V28, Amount):
    data = np.column_stack(
        [V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11, V12, V13, V14, V15, V16, V17, V18, V19, V20, V21, V22, V23, V24, V25, V26, V27, V28, Amount])
    numerical = np.column_stack(data.T)
    X = scaler.transform(numerical)
    return model.predict(xgb.DMatrix(X))


con.create_function("udf", udf,
                    [DOUBLE, DOUBLE, DOUBLE, DOUBLE, DOUBLE, DOUBLE, DOUBLE, DOUBLE, DOUBLE, DOUBLE, DOUBLE, DOUBLE,
                     DOUBLE, DOUBLE, DOUBLE, DOUBLE, DOUBLE, DOUBLE, DOUBLE, DOUBLE, DOUBLE, DOUBLE, DOUBLE,
                     DOUBLE, DOUBLE, DOUBLE, DOUBLE, DOUBLE, DOUBLE], BIGINT, type="arrow")

# con.sql("SET threads TO 1;")

con.sql('''
Explain analyze SELECT Time, Amount, udf(V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11, V12, V13, V14, V15,
 V16, V17, V18, V19, V20, V21, V22, V23, V24, V25, V26, V27, V28, Amount) AS Class FROM Credit_Card_extension 
 WHERE V1 > 1 AND V2 < 0.27 AND V3 > 0.3;
''')

print(name)
