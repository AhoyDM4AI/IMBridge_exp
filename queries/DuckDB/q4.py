import pickle

import duckdb
import numpy as np
from duckdb.typing import BIGINT, DOUBLE, FLOAT, VARCHAR
import pandas as pd
import time
import onnxruntime as ort

name = "pf4"

con = duckdb.connect("imbridge2.db")

onnx_path = './test_raven/Flights/flights_rf_pipeline.onnx'
ortconfig = ort.SessionOptions()
flights_onnx_session = ort.InferenceSession(onnx_path, sess_options=ortconfig)
flights_label = flights_onnx_session.get_outputs()[0]
numerical_columns = ['slatitude', 'slongitude', 'dlatitude', 'dlongitude']
categorical_columns = ['name1', 'name2', 'name4', 'acountry', 'active', 'scity', 'scountry', 'stimezone', 'sdst',
                       'dcity', 'dcountry', 'dtimezone', 'ddst']
flights_input_columns = numerical_columns + categorical_columns
flights_type_map = {
    'int32': np.int64,
    'int64': np.int64,
    'float64': np.float32,
    'object': str,
}


def udf(slatitude, slongitude, dlatitude, dlongitude, name1, name2, name4, acountry, active,
        scity, scountry, stimezone, sdst, dcity, dcountry, dtimezone, ddst):
    def udf_wrap(*args):
        infer_batch = {
            elem: args[i].to_numpy().astype(flights_type_map[args[i].to_numpy().dtype.name]).reshape((-1, 1))
            for i, elem in enumerate(flights_input_columns)
        }
        outputs = flights_onnx_session.run([flights_label.name], infer_batch)
        return outputs[0]

    return udf_wrap(slatitude, slongitude, dlatitude, dlongitude, name1, name2, name4, acountry, active,
                    scity, scountry, stimezone, sdst, dcity, dcountry, dtimezone, ddst)


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