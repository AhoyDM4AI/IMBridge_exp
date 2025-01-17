import dycacher
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, LabelBinarizer
import torch
from torch import nn
import duckdb
from duckdb.typing import BIGINT, DOUBLE, FLOAT, VARCHAR
import pandas as pd

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(None, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 5)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.softmax(self.fc3(x))
        return x
    
name = "q4 tpch q5"

con = duckdb.connect("imbridge_tpch_10.db")

def udf(c_acctbal, o_totalprice, l_quantity, l_extendedprice, l_discount, l_tax, s_acctbal,
  o_orderstatus, l_returnflag, l_linestatus, l_shipinstruct, 
  l_shipmode, n_nationkey, n_regionkey):
    scaler_path = '/home/test_tpch/Q5_standard_scale_model.pkl'
    enc_path = '/home/test_tpch/Q5_one_hot_encoder.pkl'
    lb_path = '/home/test_tpch/Q5_label_binarizer.pkl'
    model_path = '/home/test_tpch/Q5_pytorch_mlp.model'
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    with open(enc_path, 'rb') as f:
        enc = pickle.load(f)
    with open(lb_path, 'rb') as f:
        lb = pickle.load(f)
    mlp = torch.load(model_path)
    mlp.eval()

    data = np.column_stack([c_acctbal, o_totalprice, l_quantity, l_extendedprice, l_discount, l_tax, s_acctbal,
  o_orderstatus, l_returnflag, l_linestatus, l_shipinstruct, 
  l_shipmode, n_nationkey, n_regionkey])
    data = np.split(data, np.array([7]), axis = 1)
    numerical = data[0]
    categorical = data[1]
    X = torch.tensor(np.hstack((scaler.transform(numerical), enc.transform(categorical).toarray())), dtype=torch.float32)
    with torch.no_grad():
        predictions = mlp(X)
    res = lb.inverse_transform(predictions.numpy())
    return res


con.create_function("udf", udf,
                    [DOUBLE, DOUBLE, DOUBLE, DOUBLE, DOUBLE, DOUBLE, DOUBLE, VARCHAR,
                      VARCHAR, VARCHAR, VARCHAR, VARCHAR, DOUBLE, DOUBLE], VARCHAR, type="arrow")

# con.sql("SET threads TO 1;")

print(con.sql('''
EXPLAIN ANALYZE
select
 n_name,
 sum(l_extendedprice * (1 - l_discount)) as revenue
from
 customer,
 orders,
 lineitem,
 supplier,
 nation,
 region
where
 c_custkey = o_custkey
 and l_orderkey = o_orderkey
 and l_suppkey = s_suppkey
 and c_nationkey = s_nationkey
 and s_nationkey = n_nationkey
 and n_regionkey = r_regionkey
 and r_name = 'ASIA'
 and o_orderdate >= date '1994-01-01'
 and o_orderdate < date '1994-01-01' + interval '1' year
 and udf(c_acctbal, o_totalprice, l_quantity, l_extendedprice, l_discount, l_tax, s_acctbal,
  o_orderstatus, l_returnflag, l_linestatus, l_shipinstruct, l_shipmode, n_nationkey, n_regionkey) = '1-URGENT'
group by
 n_name
order by
 revenue desc;
'''))

print(name)