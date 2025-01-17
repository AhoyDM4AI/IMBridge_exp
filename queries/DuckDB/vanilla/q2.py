import pickle

import duckdb
import numpy as np
from duckdb.typing import BIGINT, DOUBLE, FLOAT, VARCHAR
import pandas as pd
import time
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

name = "q2"

con = duckdb.connect("imbridge2.db")


def udf(slatitude, slongitude, dlatitude, dlongitude, name1, name2, name4, acountry, active,
 scity, scountry, stimezone, sdst, dcity, dcountry, dtimezone, ddst):
    
    scaler_path = '/home/test_raven/Flights/flights_standard_scale_model.pkl'
    enc_path = '/home/test_raven/Flights/flights_one_hot_encoder.pkl'
    model_path = '/home/test_raven/Flights/flights_rf_model.pkl'
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    with open(enc_path, 'rb') as f:
        enc = pickle.load(f)
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    data = np.column_stack([slatitude, slongitude, dlatitude, dlongitude, name1, name2, name4, acountry, active,
        scity, scountry, stimezone, sdst, dcity, dcountry, dtimezone, ddst])
    data = np.split(data, np.array([4]), axis=1)
    numerical = data[0]
    categorical = data[1]

    X = np.hstack((scaler.transform(numerical), enc.transform(categorical).toarray()))
    return model.predict(X)


con.create_function("udf", udf,
                    [DOUBLE, DOUBLE, DOUBLE, DOUBLE, BIGINT, VARCHAR, VARCHAR, VARCHAR, VARCHAR, VARCHAR, VARCHAR,
                     BIGINT, VARCHAR, VARCHAR, VARCHAR, BIGINT, VARCHAR], BIGINT, type="arrow")

#con.sql("SET threads TO 1;")

con.sql('''
Explain analyze SELECT Flights_S_routes_extension2.airlineid, Flights_S_routes_extension2.sairportid, Flights_S_routes_extension2.dairportid,
 udf(slatitude, slongitude, dlatitude, dlongitude, name1, name2, name4, acountry, active, 
 scity, scountry, stimezone, sdst, dcity, dcountry, dtimezone, ddst) AS codeshare 
 FROM Flights_S_routes_extension2 JOIN Flights_R1_airlines2 ON Flights_S_routes_extension2.airlineid = Flights_R1_airlines2.airlineid 
 JOIN Flights_R2_sairports ON Flights_S_routes_extension2.sairportid = Flights_R2_sairports.sairportid JOIN Flights_R3_dairports 
 ON Flights_S_routes_extension2.dairportid = Flights_R3_dairports.dairportid
WHERE name2 = 't' and name4 = 't' and name1 > 2.8;
''')

print(name)
