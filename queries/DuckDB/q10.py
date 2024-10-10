import duckdb
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

con = duckdb.connect("imbridge.db")

name = "q10"
mname = "uc06"
model_file_name = f"/home/model/{mname}/{mname}.python.model"

model = joblib.load(model_file_name)


def udf(smart_5_raw,
        smart_10_raw,
        smart_184_raw,
        smart_187_raw,
        smart_188_raw,
        smart_197_raw,
        smart_198_raw):
    data = pd.DataFrame({
        'smart_5_raw': smart_5_raw,
        'smart_10_raw': smart_10_raw,
        'smart_184_raw': smart_184_raw,
        'smart_187_raw': smart_187_raw,
        'smart_188_raw': smart_188_raw,
        'smart_197_raw': smart_197_raw,
        'smart_198_raw': smart_198_raw
    })
    return model.predict(data)


con.create_function("udf", udf, [DOUBLE, DOUBLE, DOUBLE, DOUBLE, DOUBLE, DOUBLE, DOUBLE], BIGINT, type="arrow")

# con.sql("SET threads TO 1;")

res = con.sql('''
explain analyze select serial_number, udf(smart_5_raw, smart_10_raw, smart_184_raw, smart_187_raw, smart_188_raw, smart_197_raw, smart_198_raw) 
from Failures;
''')

print(name)
