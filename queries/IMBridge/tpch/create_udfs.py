import pymysql
import json
import time
import pandas as pd
import os

trace_on = "SET ob_enable_show_trace = 1;"
show_trace = "SHOW TRACE;"
plan_flush = "ALTER SYSTEM FLUSH PLAN CACHE;"

prefix = "/home/test/experiments"

output_path = "./create_udfs.log"

def run_sql(cur, sql):
	cur.execute(plan_flush)
	cur.execute(sql)
	#time_consuming = analysis_trace(cur)
	#return time_consuming
	rows = cur.fetchall()
	for row in rows:
		print(row)

def analysis_trace(cur):
	cur.execute(show_trace)
	trace = cur.fetchone()
	if trace is not None:
		return trace[2]
	else:
		return -1

# raven
# expedia + sklearn + decision tree
orderpriority_pytorch_mlp_unopt = '''
CREATE PYTHON_UDF orderpriority_pytorch_mlp_unopt(c_acctbal INTEGER, o_totalprice INTEGER, 
  l_quantity INTEGER, l_extendedprice INTEGER, l_discount INTEGER, l_tax INTEGER, s_acctbal INTEGER,
  o_orderstatus STRING, l_returnflag STRING, l_linestatus STRING, l_shipinstruct STRING, 
  l_shipmode STRING, n_nationkey INTEGER, n_regionkey INTEGER) RETURNS STRING {{"
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, LabelBinarizer
import torch
from torch import nn
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(X_train.shape[1], 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 5)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.softmax(self.fc3(x))
        return x
def pyinitial():
    pass
def pyfun(*args):
    scaler_path = '/home/TPC-H_V3.0.1/dbgen/tbls/Q5_standard_scale_model.pkl'
    enc_path = '/home/TPC-H_V3.0.1/dbgen/tbls/Q5_one_hot_encoder.pkl'
    lb_path = '/home/TPC-H_V3.0.1/dbgen/tbls/Q5_label_binarizer.pkl'
    model_path = '/home/TPC-H_V3.0.1/dbgen/tbls/Q5_pytorch_mlp.model'
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    with open(enc_path, 'rb') as f:
        enc = pickle.load(f)
    with open(lb_path, 'rb') as f:
        lb = pickle.load(f)
    mlp = torch.load(model_path)
    mlp.eval()
    data = np.column_stack(args)
    data = np.split(data, np.array([7]), axis = 1)
    numerical = data[0]
    categorical = data[1]
    X = torch.tensor(np.hstack((scaler.transform(numerical), enc.transform(categorical).toarray())), dtype=torch.float32)
    with torch.no_grad():
        predictions = mlp(X)
    return lb.inverse_transform(predictions.numpy())
"};CREATE PYTHON_UDF orderpriority_pytorch_mlp_unopt(c_acctbal INTEGER, o_totalprice INTEGER, 
  l_quantity INTEGER, l_extendedprice INTEGER, l_discount INTEGER, l_tax INTEGER, s_acctbal INTEGER,
  o_orderstatus STRING, l_returnflag STRING, l_linestatus STRING, l_shipinstruct STRING, 
  l_shipmode STRING, n_nationkey INTEGER, n_regionkey INTEGER) RETURNS STRING {"
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, LabelBinarizer
import torch
from torch import nn
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(X_train.shape[1], 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 5)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.softmax(self.fc3(x))
        return x
def pyinitial():
    pass
def pyfun(*args):
    scaler_path = '/home/TPC-H_V3.0.1/dbgen/tbls/Q5_standard_scale_model.pkl'
    enc_path = '/home/TPC-H_V3.0.1/dbgen/tbls/Q5_one_hot_encoder.pkl'
    lb_path = '/home/TPC-H_V3.0.1/dbgen/tbls/Q5_label_binarizer.pkl'
    model_path = '/home/TPC-H_V3.0.1/dbgen/tbls/Q5_pytorch_mlp.model'
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    with open(enc_path, 'rb') as f:
        enc = pickle.load(f)
    with open(lb_path, 'rb') as f:
        lb = pickle.load(f)
    mlp = torch.load(model_path)
    mlp.eval()
    data = np.column_stack(args)
    data = np.split(data, np.array([7]), axis = 1)
    numerical = data[0]
    categorical = data[1]
    X = torch.tensor(np.hstack((scaler.transform(numerical), enc.transform(categorical).toarray())), dtype=torch.float32)
    with torch.no_grad():
        predictions = mlp(X)
    return lb.inverse_transform(predictions.numpy())
"}};
'''

returnflag_lightgbm_gdbt_unopt = '''
CREATE PYTHON_UDF returnflag_lightgbm_gdbt_unopt(c_acctbal INTEGER, o_totalprice INTEGER, 
  l_quantity INTEGER, l_extendedprice INTEGER, l_discount INTEGER, l_tax INTEGER, 
  o_orderstatus STRING, o_orderpriority STRING, l_linestatus STRING, l_shipinstruct STRING, 
  l_shipmode STRING, n_nationkey INTEGER, n_regionkey INTEGER) RETURNS STRING {{"
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, LabelBinarizer
import lightgbm as lgb
def pyinitial():
    pass
def pyfun(*args):
    scaler_path = '/home/TPC-H_V3.0.1/dbgen/tbls/Q10_standard_scale_model.pkl'
    enc_path = '/home/TPC-H_V3.0.1/dbgen/tbls/Q10_one_hot_encoder.pkl'
    lb_path = '/home/TPC-H_V3.0.1/dbgen/tbls/Q10_label_binarizer.pkl'
    model_path = '/home/TPC-H_V3.0.1/dbgen/tbls/Q10_lgb_gbdt_model.txt'
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    with open(enc_path, 'rb') as f:
        enc = pickle.load(f)
    with open(lb_path, 'rb') as f:
        lb = pickle.load(f)
    model = lgb.Booster(model_file=model_path)
    data = np.column_stack(args)
    data = np.split(data, np.array([6]), axis = 1)
    numerical = data[0]
    categorical = data[1]
    X = np.hstack((scaler.transform(numerical), enc.transform(categorical).toarray()))
    return lb.inverse_transform(model.predict(X))
"}};'''


def set_query_tpch(test_sql):
  test_sql.append(orderpriority_pytorch_mlp_unopt)
  test_sql.append(returnflag_lightgbm_gdbt_unopt)
  
drop_udf = '''DROP PYTHON_UDF {};'''


def set_query_drop_udf_tpch(sql):
  sql.append(drop_udf.format("orderpriority_pytorch_mlp_unopt")) # Q5
  sql.append(drop_udf.format("returnflag_lightgbm_gdbt_unopt")) # Q10
  
output_path = "./create_udfs.log"

# tpcx_ai connection
conn = pymysql.connect(host="127.0.0.1", port=2881, user="root", passwd="ulAFVBT0D4GDfMizqJXJ", db="tpcx_ai")
cur = conn.cursor()

try:
  test_sql = []
  time_stat = 0
  cur.execute(trace_on) # open trace
  set_query_tpch(test_sql)
  # set_query_drop_udf_tpch
  for i in test_sql:
      time_consuming = run_sql(cur, i)
      df = pd.DataFrame({'query': [i], 'execute': [time_consuming], 'time': [time.asctime()]})
      df.to_csv(output_path, index=True, mode='a', header=None)

finally:
  cur.close()
  conn.close()