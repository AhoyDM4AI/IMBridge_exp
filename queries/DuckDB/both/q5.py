import dycacher
import pickle

import duckdb
import numpy as np
from duckdb.typing import BIGINT, DOUBLE, FLOAT, VARCHAR
import pandas as pd
import time
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, LabelBinarizer
import lightgbm as lgb

name = "q5 tpch q10"

con = duckdb.connect("imbridge_tpch_10.db")


def udf(c_acctbal, o_totalprice, l_quantity, l_extendedprice, l_discount, l_tax,
         o_orderstatus, o_orderpriority, l_linestatus,
           l_shipinstruct, l_shipmode, n_nationkey, n_regionkey):
    scaler_path = '/home/test_tpch/Q10_standard_scale_model.pkl'
    enc_path = '/home/test_tpch/Q10_one_hot_encoder.pkl'
    lb_path = '/home/test_tpch/Q10_label_binarizer.pkl'
    model_path = '/home/test_tpch/Q10_lgb_gbdt_model.txt'
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    with open(enc_path, 'rb') as f:
        enc = pickle.load(f)
    with open(lb_path, 'rb') as f:
        lb = pickle.load(f)

    model = lgb.Booster(model_file=model_path)
    data = np.column_stack([c_acctbal, o_totalprice, l_quantity, l_extendedprice, l_discount, l_tax,
         o_orderstatus, o_orderpriority, l_linestatus,
           l_shipinstruct, l_shipmode, n_nationkey, n_regionkey])
    data = np.split(data, np.array([6]), axis = 1)
    numerical = data[0]
    categorical = data[1]
    X = np.hstack((scaler.transform(numerical), enc.transform(categorical).toarray()))
    res = lb.inverse_transform(model.predict(X))
    return res


con.create_function("udf", udf,
                    [DOUBLE, DOUBLE, DOUBLE, DOUBLE, DOUBLE, DOUBLE, VARCHAR,
                      VARCHAR, VARCHAR, VARCHAR, VARCHAR, DOUBLE, DOUBLE], VARCHAR, type="arrow", kind=duckdb.functional.PREDICTION, batch_size=4096)

#con.sql("SET threads TO 1;")

print(con.sql('''EXPLAIN ANALYZE 
SELECT c_custkey,
               c_name,
               sum(l_extendedprice * (1 - l_discount)) as revenue,
               c_acctbal,
               n_name,
               c_address,
               c_phone,
               c_comment
       from customer, orders, lineitem, nation
       where c_custkey = o_custkey and
             l_orderkey = o_orderkey and
             o_orderdate >= DATE'1993-10-01' and
             o_orderdate < DATE'1993-10-01' + interval '3' month and
             c_nationkey = n_nationkey and 
             udf(c_acctbal, o_totalprice, l_quantity, l_extendedprice, l_discount, l_tax, 
                   o_orderstatus, o_orderpriority, l_linestatus, l_shipinstruct, l_shipmode, n_nationkey, n_regionkey) = 'R'
       group by c_custkey, c_name, c_acctbal, c_phone, n_name, c_address, c_comment
       order by revenue desc;
'''))

print(name)
