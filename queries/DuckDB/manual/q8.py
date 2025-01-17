import duckdb
import joblib
import numpy as np
import pandas as pd
from duckdb.typing import BIGINT, DOUBLE, FLOAT, VARCHAR
import joblib
from surprise import SVD
from surprise import Dataset
from surprise.reader import Reader

con = duckdb.connect("imbridge.db")

name = "q8"
mname = "uc07"
model_file_name = f"/home/model/{mname}/{mname}.python.model"

model = joblib.load(model_file_name)


def udf(user_id, item_id):
    ratings = []

    for i in range(len(user_id)):
        rating = model.predict(user_id[i], item_id[i]).est
        ratings.append(rating)
    return np.array(ratings)


con.create_function("udf", udf, [BIGINT, BIGINT], DOUBLE, type="arrow")

#con.sql("SET threads TO 1;")

res = con.sql('''
explain analyze select userID, productID, r, score 
from (select userID, productID, score, rank() OVER (PARTITION BY userID ORDER BY score) as r 
from (select userID, productID, udf(userID, productID) score 
from (select userID, productID 
from Product_Rating
group by userID, productID)))
where r <=10;
''')

print(name)
