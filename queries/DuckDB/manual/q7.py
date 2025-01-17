import duckdb
import joblib
import numpy as np
import pandas as pd
from duckdb.typing import BIGINT, DOUBLE, FLOAT, VARCHAR
from sklearn import feature_extraction, naive_bayes
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer

con = duckdb.connect("imbridge.db")

name = "q7"
mname = "uc04"
model_file_name = f"/home/model/{mname}/{mname}.python.model"

model = joblib.load(model_file_name)


def udf(txt):
    data = pd.DataFrame({
        "text": txt
    })
    return model.predict(data["text"])


con.create_function("udf", udf, [VARCHAR], BIGINT, type="arrow")

# con.sql("SET threads TO 1;")

res = con.sql('''
explain analyze select ID, udf(txt) from 
(select ID, text txt from Review);
''')

print(name)
