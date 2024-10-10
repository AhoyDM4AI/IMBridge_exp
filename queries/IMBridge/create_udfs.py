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
expedia_sklearn_dt_init = '''CREATE PYTHON_UDF expedia_sklearn_dt_init(prop_location_score1 REAL, prop_location_score2 REAL, prop_log_historical_price REAL, price_usd REAL, orig_destination_distance REAL, prop_review_score REAL, avg_bookings_usd REAL, stdev_bookings_usd REAL, position_ STRING, prop_country_id STRING, prop_starrating INTEGER, prop_brand_bool INTEGER, count_clicks INTEGER, count_bookings INTEGER, year_ STRING, month_ STRING, weekofyear_ STRING, time_ STRING, site_id STRING, visitor_location_country_id STRING, srch_destination_id STRING, srch_length_of_stay INTEGER, srch_booking_window INTEGER, srch_adults_count INTEGER, srch_children_count INTEGER, srch_room_count INTEGER, srch_saturday_night_bool INTEGER, random_bool INTEGER) RETURNS INTEGER 
{{"
import pickle
import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
def pyinitial():
    start = time.perf_counter()
    global expedia_scaler, expedia_enc, expedia_model
    scaler_path = '{0}/test_raven/Expedia/expedia_standard_scale_model.pkl'
    enc_path = '{0}/test_raven/Expedia/expedia_one_hot_encoder.pkl'
    model_path = '{0}/test_raven/Expedia/expedia_dt_model.pkl'
    with open(scaler_path, 'rb') as f:
        expedia_scaler = pickle.load(f)
    with open(enc_path, 'rb') as f:
        expedia_enc = pickle.load(f)
    with open(model_path, 'rb') as f:
        expedia_model = pickle.load(f)
    stop = time.perf_counter()
    with open('/home/test/experiments/oceanbase/opt/both_pps.log', 'a+') as log:
        log.write(str(stop-start)+' ')
def pyfun(*args):
    data = np.column_stack(args)
    data = np.split(data, np.array([8]), axis = 1)
    numerical = data[0]
    categorical = data[1]
    X = np.hstack((expedia_scaler.transform(numerical), expedia_enc.transform(categorical).toarray()))
    return expedia_model.predict(X)
"}};
'''

expedia_tensorflow_tf_init = '''CREATE PYTHON_UDF expedia_tensorflow_df_predict(prop_location_score1 REAL, prop_location_score2 REAL, prop_log_historical_price REAL, price_usd REAL, orig_destination_distance REAL, prop_review_score REAL, avg_bookings_usd REAL, stdev_bookings_usd REAL, position_ STRING, prop_country_id STRING, prop_starrating INTEGER, prop_brand_bool INTEGER, count_clicks INTEGER, count_bookings INTEGER, year_ STRING, month_ STRING, weekofyear_ STRING, time_ STRING, site_id STRING, visitor_location_country_id STRING, srch_destination_id STRING, srch_length_of_stay INTEGER, srch_booking_window INTEGER, srch_adults_count INTEGER, srch_children_count INTEGER, srch_room_count INTEGER, srch_saturday_night_bool INTEGER, random_bool INTEGER) RETURNS INTEGER 
{{"
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

def pyinitial():
    global expedia_scaler, expedia_enc, expedia_model
    scaler_path = '{0}/test_raven/Expedia/expedia_standard_scale_model.pkl'
    enc_path = '{0}/test_raven/Expedia/expedia_one_hot_encoder.pkl'
    model_path = '{0}/test_raven/Expedia/expedia_tensorflow_dt_model.h5'
    with open(scaler_path, 'rb') as f:
        expedia_scaler = pickle.load(f)
    with open(enc_path, 'rb') as f:
        expedia_enc = pickle.load(f)
    expedia_model = tf.keras.models.load_model(model_path)

def pyfun(*args):
    data = np.column_stack(args)
    data = np.split(data, np.array([8]), axis = 1)
    numerical = data[0]
    categorical = data[1]
    X = np.hstack((expedia_scaler.transform(numerical), expedia_enc.transform(categorical).toarray()))
    return expedia_model.predict(X).flatten().astype(int)[0]
"}};'''

expedia_sklearn_dt_uninit = '''CREATE PYTHON_UDF expedia_sklearn_dt_uninit(prop_location_score1 REAL, prop_location_score2 REAL, prop_log_historical_price REAL, price_usd REAL, orig_destination_distance REAL, prop_review_score REAL, avg_bookings_usd REAL, stdev_bookings_usd REAL, position_ STRING, prop_country_id STRING, prop_starrating INTEGER, prop_brand_bool INTEGER, count_clicks INTEGER, count_bookings INTEGER, year_ STRING, month_ STRING, weekofyear_ STRING, time_ STRING, site_id STRING, visitor_location_country_id STRING, srch_destination_id STRING, srch_length_of_stay INTEGER, srch_booking_window INTEGER, srch_adults_count INTEGER, srch_children_count INTEGER, srch_room_count INTEGER, srch_saturday_night_bool INTEGER, random_bool INTEGER) RETURNS INTEGER 
{{"
import pickle
import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
def pyinitial():
    pass
def pyfun(*args):
    start = time.perf_counter()
    scaler_path = '{0}/test_raven/Expedia/expedia_standard_scale_model.pkl'
    enc_path = '{0}/test_raven/Expedia/expedia_one_hot_encoder.pkl'
    model_path = '{0}/test_raven/Expedia/expedia_dt_model.pkl'
    with open(scaler_path, 'rb') as f:
        expedia_scaler = pickle.load(f)
    with open(enc_path, 'rb') as f:
        expedia_enc = pickle.load(f)
    with open(model_path, 'rb') as f:
        expedia_model = pickle.load(f)
    stop = time.perf_counter()
    with open('/home/test/experiments/oceanbase/opt/both_pps.log', 'a+') as log:
        log.write(str(stop-start)+' ')
    data = np.column_stack(args)
    data = np.split(data, np.array([8]), axis = 1)
    numerical = data[0]
    categorical = data[1]
    X = np.hstack((expedia_scaler.transform(numerical), expedia_enc.transform(categorical).toarray()))
    return expedia_model.predict(X)
"}};
'''

expedia_onnx_dt_init = '''CREATE PYTHON_UDF expedia_onnx_dt_init(prop_location_score1 REAL, prop_location_score2 REAL, prop_log_historical_price REAL, price_usd REAL, orig_destination_distance REAL, prop_review_score REAL, avg_bookings_usd REAL, stdev_bookings_usd REAL, position_ STRING, prop_country_id STRING, prop_starrating INTEGER, prop_brand_bool INTEGER, count_clicks INTEGER, count_bookings INTEGER, year_ STRING, month_ STRING, weekofyear_ STRING, time_ STRING, site_id STRING, visitor_location_country_id STRING, srch_destination_id STRING, srch_length_of_stay INTEGER, srch_booking_window INTEGER, srch_adults_count INTEGER, srch_children_count INTEGER, srch_room_count INTEGER, srch_saturday_night_bool INTEGER, random_bool INTEGER) RETURNS INTEGER 
{{"
import time
import numpy as np
import pandas as pd
import onnxruntime as ort
def pyinitial():
    global expedia_onnx_session, expedia_type_map, expedia_input_columns, expedia_label
    onnx_path = '{0}/test_raven/Expedia/expedia_dt_pipeline.onnx'
    ortconfig = ort.SessionOptions()
    expedia_onnx_session = ort.InferenceSession(onnx_path, sess_options=ortconfig)
    expedia_label = expedia_onnx_session.get_outputs()[0]
    numerical_columns = ['prop_location_score1', 'prop_location_score2', 'prop_log_historical_price', 'price_usd', 'orig_destination_distance', 'prop_review_score', 'avg_bookings_usd', 'stdev_bookings_usd']
    categorical_columns = ['position', 'prop_country_id', 'prop_starrating', 'prop_brand_bool', 'count_clicks','count_bookings', 'year', 'month', 'weekofyear', 'time', 'site_id', 'visitor_location_country_id', 'srch_destination_id', 'srch_length_of_stay', 'srch_booking_window', 'srch_adults_count', 'srch_children_count', 'srch_room_count', 'srch_saturday_night_bool', 'random_bool']
    expedia_input_columns = numerical_columns + categorical_columns
    expedia_type_map = {{
    'int32': np.int64,
    'int64': np.int64,
    'float64': np.float32,
    'object': str,
    }}
def pyfun(*args):
    infer_batch = {{
        elem: args[i].astype(expedia_type_map[args[i].dtype.name]).reshape((-1, 1))
        for i, elem in enumerate(expedia_input_columns)
    }}
    outputs = expedia_onnx_session.run([expedia_label.name], infer_batch)
    return outputs[0]
"}};
'''

expedia_onnx_dt_uninit = '''CREATE PYTHON_UDF expedia_onnx_dt_uninit(prop_location_score1 REAL, prop_location_score2 REAL, prop_log_historical_price REAL, price_usd REAL, orig_destination_distance REAL, prop_review_score REAL, avg_bookings_usd REAL, stdev_bookings_usd REAL, position_ STRING, prop_country_id STRING, prop_starrating INTEGER, prop_brand_bool INTEGER, count_clicks INTEGER, count_bookings INTEGER, year_ STRING, month_ STRING, weekofyear_ STRING, time_ STRING, site_id STRING, visitor_location_country_id STRING, srch_destination_id STRING, srch_length_of_stay INTEGER, srch_booking_window INTEGER, srch_adults_count INTEGER, srch_children_count INTEGER, srch_room_count INTEGER, srch_saturday_night_bool INTEGER, random_bool INTEGER) RETURNS INTEGER 
{{"
import time
import numpy as np
import pandas as pd
import onnxruntime as ort
def pyinitial():
    pass
def pyfun(*args):
    onnx_path = '{0}/test_raven/Expedia/expedia_dt_pipeline.onnx'
    ortconfig = ort.SessionOptions()
    onnx_session = ort.InferenceSession(onnx_path, sess_options=ortconfig)
    label = onnx_session.get_outputs()[0]
    numerical_columns = ['prop_location_score1', 'prop_location_score2', 'prop_log_historical_price', 'price_usd', 'orig_destination_distance', 'prop_review_score', 'avg_bookings_usd', 'stdev_bookings_usd']
    categorical_columns = ['position', 'prop_country_id', 'prop_starrating', 'prop_brand_bool', 'count_clicks','count_bookings', 'year', 'month', 'weekofyear', 'time', 'site_id', 'visitor_location_country_id', 'srch_destination_id', 'srch_length_of_stay', 'srch_booking_window', 'srch_adults_count', 'srch_children_count', 'srch_room_count', 'srch_saturday_night_bool', 'random_bool']
    input_columns = numerical_columns + categorical_columns
    type_map = {{
    'int32': np.int64,
    'int64': np.int64,
    'float64': np.float32,
    'object': str,
    }}
    infer_batch = {{
        elem: args[i].astype(type_map[args[i].dtype.name]).reshape((-1, 1))
        for i, elem in enumerate(input_columns)
    }}
    outputs = onnx_session.run([label.name], infer_batch)
    return outputs[0]
"}};
'''

# flights + onnx + random forest
flights_onnx_rf_init = '''CREATE PYTHON_UDF flights_onnx_rf_init(slatitude REAL, slongitude REAL, dlatitude REAL, dlongitude REAL, name1 INTEGER, name2 STRING, name4 STRING, acountry STRING, active_ STRING, scity STRING, scountry STRING, stimezone INTEGER, sdst STRING, dcity STRING, dcountry STRING, dtimezone INTEGER, ddst STRING) RETURNS INTEGER 
{{"
import numpy as np
import pandas as pd
import onnxruntime as ort
def pyinitial():
    global flights_onnx_session, flights_type_map, flights_input_columns, flights_label
    onnx_path = '{}/test_raven/Flights/flights_rf_pipeline.onnx'
    ortconfig = ort.SessionOptions()
    flights_onnx_session = ort.InferenceSession(onnx_path, sess_options=ortconfig)
    flights_label = flights_onnx_session.get_outputs()[0]
    numerical_columns = ['slatitude', 'slongitude', 'dlatitude', 'dlongitude']
    categorical_columns = ['name1', 'name2', 'name4', 'acountry', 'active', 'scity', 'scountry', 'stimezone', 'sdst', 'dcity', 'dcountry', 'dtimezone', 'ddst']
    flights_input_columns = numerical_columns + categorical_columns
    flights_type_map = {{
    'int32': np.int64,
    'int64': np.int64,
    'float64': np.float32,
    'object': str,
    }}
def pyfun(*args):
    infer_batch = {{
        elem: args[i].astype(flights_type_map[args[i].dtype.name]).reshape((-1, 1))
        for i, elem in enumerate(flights_input_columns)
    }}
    outputs = flights_onnx_session.run([flights_label.name], infer_batch)
    return outputs[0]
"}};
'''

flights_onnx_rf_uninit = '''CREATE PYTHON_UDF flights_onnx_rf_uninit(slatitude REAL, slongitude REAL, dlatitude REAL, dlongitude REAL, name1 INTEGER, name2 STRING, name4 STRING, acountry STRING, active_ STRING, scity STRING, scountry STRING, stimezone INTEGER, sdst STRING, dcity STRING, dcountry STRING, dtimezone INTEGER, ddst STRING) RETURNS INTEGER 
{{"
import numpy as np
import pandas as pd
import onnxruntime as ort
def pyinitial():
    pass
def pyfun(*args):
    onnx_path = '{}/test_raven/Flights/flights_rf_pipeline.onnx'
    ortconfig = ort.SessionOptions()
    flights_onnx_session = ort.InferenceSession(onnx_path, sess_options=ortconfig)
    flights_label = flights_onnx_session.get_outputs()[0]
    numerical_columns = ['slatitude', 'slongitude', 'dlatitude', 'dlongitude']
    categorical_columns = ['name1', 'name2', 'name4', 'acountry', 'active', 'scity', 'scountry', 'stimezone', 'sdst', 'dcity', 'dcountry', 'dtimezone', 'ddst']
    flights_input_columns = numerical_columns + categorical_columns
    flights_type_map = {{
    'int32': np.int64,
    'int64': np.int64,
    'float64': np.float32,
    'object': str,
    }}
    infer_batch = {{
        elem: args[i].astype(flights_type_map[args[i].dtype.name]).reshape((-1, 1))
        for i, elem in enumerate(flights_input_columns)
    }}
    outputs = flights_onnx_session.run([flights_label.name], infer_batch)
    return outputs[0]
"}};
'''

flights_sklearn_rf_init = '''CREATE PYTHON_UDF flights_sklearn_rf_init(slatitude REAL, slongitude REAL, dlatitude REAL, dlongitude REAL, name1 INTEGER, name2 STRING, name4 STRING, acountry STRING, active_ STRING, scity STRING, scountry STRING, stimezone INTEGER, sdst STRING, dcity STRING, dcountry STRING, dtimezone INTEGER, ddst STRING) RETURNS INTEGER 
{{"
import pickle
import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
def pyinitial():
    global flights_scaler, flights_enc, flights_model
    scaler_path = '{0}/test_raven/Flights/flights_standard_scale_model.pkl'
    enc_path = '{0}/test_raven/Flights/flights_one_hot_encoder.pkl'
    model_path = '{0}/test_raven/Flights/flights_rf_model.pkl'
    with open(scaler_path, 'rb') as f:
        flights_scaler = pickle.load(f)
    with open(enc_path, 'rb') as f:
        flights_enc = pickle.load(f)
    with open(model_path, 'rb') as f:
        flights_model = pickle.load(f)
def pyfun(*args):
    data = np.column_stack(args)
    data = np.split(data, np.array([4]), axis = 1)
    numerical = data[0]
    categorical = data[1]
    X = np.hstack((flights_scaler.transform(numerical), flights_enc.transform(categorical).toarray()))
    return flights_model.predict(X)
"}};
'''

flights_sklearn_rf_uninit = '''CREATE PYTHON_UDF flights_sklearn_rf_uninit(slatitude REAL, slongitude REAL, dlatitude REAL, dlongitude REAL, name1 INTEGER, name2 STRING, name4 STRING, acountry STRING, active_ STRING, scity STRING, scountry STRING, stimezone INTEGER, sdst STRING, dcity STRING, dcountry STRING, dtimezone INTEGER, ddst STRING) RETURNS INTEGER 
{{"
import pickle
import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
def pyinitial():
    pass
def pyfun(*args):
    scaler_path = '{0}/test_raven/Flights/flights_standard_scale_model.pkl'
    enc_path = '{0}/test_raven/Flights/flights_one_hot_encoder.pkl'
    model_path = '{0}/test_raven/Flights/flights_rf_model.pkl'
    with open(scaler_path, 'rb') as f:
        flights_scaler = pickle.load(f)
    with open(enc_path, 'rb') as f:
        flights_enc = pickle.load(f)
    with open(model_path, 'rb') as f:
        flights_model = pickle.load(f)
    data = np.column_stack(args)
    data = np.split(data, np.array([4]), axis = 1)
    numerical = data[0]
    categorical = data[1]
    X = np.hstack((flights_scaler.transform(numerical), flights_enc.transform(categorical).toarray()))
    return flights_model.predict(X)
"}};
'''

flights_sklearn_rf_uninit_trees = '''CREATE PYTHON_UDF flights_sklearn_rf_uninit_trees{1}(slatitude REAL, slongitude REAL, dlatitude REAL, dlongitude REAL, name1 INTEGER, name2 STRING, name4 STRING, acountry STRING, active_ STRING, scity STRING, scountry STRING, stimezone INTEGER, sdst STRING, dcity STRING, dcountry STRING, dtimezone INTEGER, ddst STRING) RETURNS INTEGER 
{{"
import pickle
import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
def pyinitial():
    pass
def pyfun(*args):
    scaler_path = '{0}/test_raven/Flights/flights_standard_scale_model.pkl'
    enc_path = '{0}/test_raven/Flights/flights_one_hot_encoder.pkl'
    model_path = '{0}/forest_models/rf/flights_rf_model_trees{1}_depth6.pkl'
    with open(scaler_path, 'rb') as f:
        flights_scaler = pickle.load(f)
    with open(enc_path, 'rb') as f:
        flights_enc = pickle.load(f)
    with open(model_path, 'rb') as f:
        flights_model = pickle.load(f)
    data = np.column_stack(args)
    data = np.split(data, np.array([4]), axis = 1)
    numerical = data[0]
    categorical = data[1]
    X = np.hstack((flights_scaler.transform(numerical), flights_enc.transform(categorical).toarray()))
    return flights_model.predict(X)
"}};
'''

# hospital + sklearn + linear regression
hospital_sklearn_lr_init = '''CREATE PYTHON_UDF hospital_sklearn_lr_init(hematocrit REAL, neutrophils REAL, sodium REAL, glucose REAL, bloodureanitro REAL, creatinine REAL, bmi REAL, pulse INTEGER, respiration REAL, secondarydiagnosisnonicd9 INTEGER, rcount STRING, gender STRING, dialysisrenalendstage INTEGER, asthma INTEGER, irondef INTEGER, pneum INTEGER, substancedependence INTEGER, psychologicaldisordermajor INTEGER, depress INTEGER, psychother INTEGER, fibrosisandother INTEGER, malnutrition INTEGER, hemo INTEGER) RETURNS INTEGER 
{{"
import pickle
import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
def pyinitial():
    global hospital_scaler, hospital_enc, hospital_model
    scaler_path = '{0}/test_raven/Hospital/hospital_standard_scale_model.pkl'
    enc_path = '{0}/test_raven/Hospital/hospital_one_hot_encoder.pkl'
    model_path = '{0}/test_raven/Hospital/hospital_lr_model.pkl'
    with open(scaler_path, 'rb') as f:
        hospital_scaler = pickle.load(f)
    with open(enc_path, 'rb') as f:
        hospital_enc = pickle.load(f)
    with open(model_path, 'rb') as f:
        hospital_model = pickle.load(f)
def pyfun(*args):
    data = np.column_stack(args)
    data = np.split(data, np.array([10]), axis = 1)
    numerical = data[0]
    categorical = data[1]
    X = np.hstack((hospital_scaler.transform(numerical), hospital_enc.transform(categorical).toarray()))
    return hospital_model.predict(X)
"}};'''

hospital_sklearn_lr_uninit = '''CREATE PYTHON_UDF hospital_sklearn_lr_uninit(hematocrit REAL, neutrophils REAL, sodium REAL, glucose REAL, bloodureanitro REAL, creatinine REAL, bmi REAL, pulse INTEGER, respiration REAL, secondarydiagnosisnonicd9 INTEGER, rcount STRING, gender STRING, dialysisrenalendstage INTEGER, asthma INTEGER, irondef INTEGER, pneum INTEGER, substancedependence INTEGER, psychologicaldisordermajor INTEGER, depress INTEGER, psychother INTEGER, fibrosisandother INTEGER, malnutrition INTEGER, hemo INTEGER) RETURNS INTEGER 
{{"
import pickle
import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
def pyinitial():
    pass
def pyfun(*args):
    scaler_path = '{0}/test_raven/Hospital/hospital_standard_scale_model.pkl'
    enc_path = '{0}/test_raven/Hospital/hospital_one_hot_encoder.pkl'
    model_path = '{0}/test_raven/Hospital/hospital_lr_model.pkl'
    with open(scaler_path, 'rb') as f:
        hospital_scaler = pickle.load(f)
    with open(enc_path, 'rb') as f:
        hospital_enc = pickle.load(f)
    with open(model_path, 'rb') as f:
        hospital_model = pickle.load(f)
    data = np.column_stack(args)
    data = np.split(data, np.array([10]), axis = 1)
    numerical = data[0]
    categorical = data[1]
    X = np.hstack((hospital_scaler.transform(numerical), hospital_enc.transform(categorical).toarray()))
    return hospital_model.predict(X)
"}};'''

# hospital + tensorflow / pytorch + mlp
hospital_tensorflow_mlp_init = '''CREATE PYTHON_UDF hospital_tensorflow_mlp_init(hematocrit REAL, neutrophils REAL, sodium REAL, glucose REAL, bloodureanitro REAL, creatinine REAL, bmi REAL, pulse INTEGER, respiration REAL, secondarydiagnosisnonicd9 INTEGER, rcount STRING, gender STRING, dialysisrenalendstage INTEGER, asthma INTEGER, irondef INTEGER, pneum INTEGER, substancedependence INTEGER, psychologicaldisordermajor INTEGER, depress INTEGER, psychother INTEGER, fibrosisandother INTEGER, malnutrition INTEGER, hemo INTEGER) RETURNS INTEGER 
{{"
import pickle
import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
def pyinitial():
    global hospital_scaler, hospital_enc, hospital_mlp_model
    scaler_path = '{0}/test_raven/Hospital/hospital_standard_scale_model.pkl'
    enc_path = '{0}/test_raven/Hospital/hospital_one_hot_encoder.pkl'
    model_path = '{0}/test_raven/Hospital/tensorflow_mlp_hospital.keras'
    with open(scaler_path, 'rb') as f:
        hospital_scaler = pickle.load(f)
    with open(enc_path, 'rb') as f:
        hospital_enc = pickle.load(f)
    hospital_mlp_model = tf.keras.models.load_model(model_path)
def pyfun(*args):
    data = np.column_stack(args)
    data = np.split(data, np.array([10]), axis = 1)
    numerical = data[0]
    categorical = data[1]
    X = np.hstack((hospital_scaler.transform(numerical), hospital_enc.transform(categorical).toarray()))
    return hospital_mlp_model.predict(X)
"}};'''


hospital_pytorch_mlp_init = '''CREATE PYTHON_UDF hospital_pytorch_mlp_init(hematocrit REAL, neutrophils REAL, sodium REAL, glucose REAL, bloodureanitro REAL, creatinine REAL, bmi REAL, pulse INTEGER, respiration REAL, secondarydiagnosisnonicd9 INTEGER, rcount STRING, gender STRING, dialysisrenalendstage INTEGER, asthma INTEGER, irondef INTEGER, pneum INTEGER, substancedependence INTEGER, psychologicaldisordermajor INTEGER, depress INTEGER, psychother INTEGER, fibrosisandother INTEGER, malnutrition INTEGER, hemo INTEGER) RETURNS INTEGER 
{{"
import numpy as np
import pandas as pd
from torch import nn
import pickle
import time
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import torch
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(27, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
def pyinitial():
    start = time.perf_counter()
    global hospital_scaler, hospital_enc, hospital_mlp_model
    scaler_path = '{0}/test_raven/Hospital/hospital_standard_scale_model.pkl'
    enc_path = '{0}/test_raven/Hospital/hospital_one_hot_encoder.pkl'
    model_path = '{0}/test_raven/Hospital/hospital_mlp_pytorch_2.model'
    with open(scaler_path, 'rb') as f:
        hospital_scaler = pickle.load(f)
    with open(enc_path, 'rb') as f:
        hospital_enc = pickle.load(f)
    hospital_mlp_model = torch.load(model_path)
    hospital_mlp_model.eval()
    stop = time.perf_counter()
    with open('/home/test/experiments/oceanbase/opt/both_pps.log', 'a+') as log:
        log.write(str(stop-start)+' ')
def pyfun(*args):
    data = np.column_stack(args)
    data = np.split(data, np.array([10]), axis = 1)
    numerical = data[0]
    categorical = data[1]
    X = torch.tensor(np.hstack((hospital_scaler.transform(numerical), hospital_enc.transform(categorical).toarray())), dtype=torch.float32)
    with torch.no_grad():
        predictions = hospital_mlp_model(X)
    return predictions.numpy()
"}};'''
hospital_pytorch_mlp_uninit = '''CREATE PYTHON_UDF hospital_pytorch_mlp_uninit(hematocrit REAL, neutrophils REAL, sodium REAL, glucose REAL, bloodureanitro REAL, creatinine REAL, bmi REAL, pulse INTEGER, respiration REAL, secondarydiagnosisnonicd9 INTEGER, rcount STRING, gender STRING, dialysisrenalendstage INTEGER, asthma INTEGER, irondef INTEGER, pneum INTEGER, substancedependence INTEGER, psychologicaldisordermajor INTEGER, depress INTEGER, psychother INTEGER, fibrosisandother INTEGER, malnutrition INTEGER, hemo INTEGER) RETURNS INTEGER 
{{"
import numpy as np
import pandas as pd
from torch import nn
import pickle
import time
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import torch
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(27, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
def pyinitial():
    pass
def pyfun(*args):
    start = time.perf_counter()
    scaler_path = '{0}/test_raven/Hospital/hospital_standard_scale_model.pkl'
    enc_path = '{0}/test_raven/Hospital/hospital_one_hot_encoder.pkl'
    model_path = '{0}/test_raven/Hospital/hospital_mlp_pytorch_2.model'
    with open(scaler_path, 'rb') as f:
        hospital_scaler = pickle.load(f)
    with open(enc_path, 'rb') as f:
        hospital_enc = pickle.load(f)
    hospital_mlp_model = torch.load(model_path)
    hospital_mlp_model.eval()
    stop = time.perf_counter()
    with open('/home/test/experiments/oceanbase/opt/both_pps.log', 'a+') as log:
        log.write(str(stop-start)+' ')
    data = np.column_stack(args)
    data = np.split(data, np.array([10]), axis = 1)
    numerical = data[0]
    categorical = data[1]
    X = torch.tensor(np.hstack((hospital_scaler.transform(numerical), hospital_enc.transform(categorical).toarray())), dtype=torch.float32)
    with torch.no_grad():
        predictions = hospital_mlp_model(X)
    return predictions.numpy()
"}};'''

# credit card + lightgbm + gradient boosting
credit_card_lightgbm_init = '''CREATE PYTHON_UDF creditcard_lightgbm_gb_init(V1 REAL, V2 REAL, V3 REAL, V4 REAL, V5 REAL, V6 REAL, V7 REAL, V8 REAL, V9 REAL, V10 REAL, V11 REAL, V12 REAL, V13 REAL, V14 REAL, V15 REAL, V16 REAL, V17 REAL, V18 REAL, V19 REAL, V20 REAL, V21 REAL, V22 REAL, V23 REAL, V24 REAL, V25 REAL, V26 REAL, V27 REAL, V28 REAL, Amount REAL) RETURNS INTEGER 
{{"
import pickle
import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
def pyinitial():
    global creditcard_scaler, creditcard_model
    scaler_path = '{0}/test_raven/Credit_Card/creditcard_standard_scale_model.pkl'
    model_path = '{0}/test_raven/Credit_Card/creditcard_lgb_model.txt'
    with open(scaler_path, 'rb') as f:
        creditcard_scaler = pickle.load(f)
    creditcard_model = lgb.Booster(model_file=model_path)
def pyfun(*args):
    numerical = np.column_stack(args)
    X = creditcard_scaler.transform(numerical)
    return creditcard_model.predict(X)
"}};'''

credit_card_lightgbm_uninit = '''CREATE PYTHON_UDF creditcard_lightgbm_gb_uninit(V1 REAL, V2 REAL, V3 REAL, V4 REAL, V5 REAL, V6 REAL, V7 REAL, V8 REAL, V9 REAL, V10 REAL, V11 REAL, V12 REAL, V13 REAL, V14 REAL, V15 REAL, V16 REAL, V17 REAL, V18 REAL, V19 REAL, V20 REAL, V21 REAL, V22 REAL, V23 REAL, V24 REAL, V25 REAL, V26 REAL, V27 REAL, V28 REAL, Amount REAL) RETURNS INTEGER 
{{"
import pickle
import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
def pyinitial():
    pass
def pyfun(*args):
    scaler_path = '{0}/test_raven/Credit_Card/creditcard_standard_scale_model.pkl'
    model_path = '{0}/test_raven/Credit_Card/creditcard_lgb_model.txt'
    with open(scaler_path, 'rb') as f:
        creditcard_scaler = pickle.load(f)
    creditcard_model = lgb.Booster(model_file=model_path)
    numerical = np.column_stack(args)
    X = creditcard_scaler.transform(numerical)
    return creditcard_model.predict(X)
"}};'''

credit_card_lightgbm_uninit_trees = '''CREATE PYTHON_UDF creditcard_lightgbm_gb_uninit_trees{1}(V1 REAL, V2 REAL, V3 REAL, V4 REAL, V5 REAL, V6 REAL, V7 REAL, V8 REAL, V9 REAL, V10 REAL, V11 REAL, V12 REAL, V13 REAL, V14 REAL, V15 REAL, V16 REAL, V17 REAL, V18 REAL, V19 REAL, V20 REAL, V21 REAL, V22 REAL, V23 REAL, V24 REAL, V25 REAL, V26 REAL, V27 REAL, V28 REAL, Amount REAL) RETURNS INTEGER 
{{"
import pickle
import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
def pyinitial():
    pass
def pyfun(*args):
    scaler_path = '{0}/test_raven/Credit_Card/creditcard_standard_scale_model.pkl'
    model_path = '{0}/forest_models/lgbm/creditcard_lgbm_model_trees{1}_depth6.json'
    with open(scaler_path, 'rb') as f:
        creditcard_scaler = pickle.load(f)
    creditcard_model = lgb.Booster(model_file=model_path)
    numerical = np.column_stack(args)
    X = creditcard_scaler.transform(numerical)
    return creditcard_model.predict(X)
"}};'''


credit_card_xgboost_init = '''CREATE PYTHON_UDF creditcard_xgboost_gb_init(V1 REAL, V2 REAL, V3 REAL, V4 REAL, V5 REAL, V6 REAL, V7 REAL, V8 REAL, V9 REAL, V10 REAL, V11 REAL, V12 REAL, V13 REAL, V14 REAL, V15 REAL, V16 REAL, V17 REAL, V18 REAL, V19 REAL, V20 REAL, V21 REAL, V22 REAL, V23 REAL, V24 REAL, V25 REAL, V26 REAL, V27 REAL, V28 REAL, Amount REAL) RETURNS INTEGER 
{{"
import pickle
import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
def pyinitial():
    global creditcard_scaler, creditcard_model
    scaler_path = '{0}/test_raven/Credit_Card/creditcard_standard_scale_model.pkl'
    model_path = '{0}/test_raven/Credit_Card/creditcard_xgb_model.json'
    with open(scaler_path, 'rb') as f:
        creditcard_scaler = pickle.load(f)
    creditcard_model = xgb.Booster(model_file=model_path)
def pyfun(*args):
    numerical = np.column_stack(args)
    X = creditcard_scaler.transform(numerical)
    return creditcard_model.predict(xgb.DMatrix(X))
"}};'''

credit_card_xgboost_uninit = '''CREATE PYTHON_UDF creditcard_xgboost_gb_uninit(V1 REAL, V2 REAL, V3 REAL, V4 REAL, V5 REAL, V6 REAL, V7 REAL, V8 REAL, V9 REAL, V10 REAL, V11 REAL, V12 REAL, V13 REAL, V14 REAL, V15 REAL, V16 REAL, V17 REAL, V18 REAL, V19 REAL, V20 REAL, V21 REAL, V22 REAL, V23 REAL, V24 REAL, V25 REAL, V26 REAL, V27 REAL, V28 REAL, Amount REAL) RETURNS INTEGER 
{{"
import pickle
import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
def pyinitial():
    pass
def pyfun(*args):
    scaler_path = '{0}/test_raven/Credit_Card/creditcard_standard_scale_model.pkl'
    model_path = '{0}/test_raven/Credit_Card/creditcard_xgb_model.json'
    with open(scaler_path, 'rb') as f:
        creditcard_scaler = pickle.load(f)
    creditcard_model = xgb.Booster(model_file=model_path)
    numerical = np.column_stack(args)
    X = creditcard_scaler.transform(numerical)
    return creditcard_model.predict(xgb.DMatrix(X))
"}};'''

credit_card_xgboost_uninit_trees = '''CREATE PYTHON_UDF creditcard_xgboost_gb_uninit_trees{1}(V1 REAL, V2 REAL, V3 REAL, V4 REAL, V5 REAL, V6 REAL, V7 REAL, V8 REAL, V9 REAL, V10 REAL, V11 REAL, V12 REAL, V13 REAL, V14 REAL, V15 REAL, V16 REAL, V17 REAL, V18 REAL, V19 REAL, V20 REAL, V21 REAL, V22 REAL, V23 REAL, V24 REAL, V25 REAL, V26 REAL, V27 REAL, V28 REAL, Amount REAL) RETURNS INTEGER 
{{"
import pickle
import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
def pyinitial():
    pass
def pyfun(*args):
    scaler_path = '{0}/test_raven/Credit_Card/creditcard_standard_scale_model.pkl'
    model_path = '{0}/forest_models/xgboost/creditcard_xgb_model_trees{1}_depth6.json'
    with open(scaler_path, 'rb') as f:
        creditcard_scaler = pickle.load(f)
    creditcard_model = xgb.Booster(model_file=model_path)
    numerical = np.column_stack(args)
    X = creditcard_scaler.transform(numerical)
    return creditcard_model.predict(xgb.DMatrix(X))
"}};'''

# tpcx_ai 
# uc01
tpcx_ai_uc01 = '''CREATE PYTHON_UDF tpcx_ai_uc01_{0}_init(return_ratio REAL, frequency_ REAL) RETURNS INTEGER 
{{"
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

def pyinitial():
    global uc01_model
    model_file = '{1}/tpcxai_datasets/{0}/model/uc01/uc01.python.model'
    uc01_model = joblib.load(model_file)
	
def pyfun(*args):
    feat = pd.DataFrame({{
        'return_ratio': args[0],
        'frequency': args[1]
    }})
    return uc01_model.predict(feat)
"}};'''

tpcx_ai_uc01_uninit = '''CREATE PYTHON_UDF tpcx_ai_uc01_{0}_uninit(return_ratio REAL, frequency_ REAL) RETURNS INTEGER 
{{"
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

def pyinitial():
    pass
	
def pyfun(*args):
    model_file = '{1}/tpcxai_datasets/{0}/model/uc01/uc01.python.model'
    uc01_model = joblib.load(model_file)
    feat = pd.DataFrame({{
        'return_ratio': args[0],
        'frequency': args[1]
    }})
    return uc01_model.predict(feat)
"}};'''

tpcx_ai_uc03 = '''CREATE PYTHON_UDF tpcx_ai_uc03_{0}_init(store INTEGER, department STRING) RETURNS STRING 
{{"
import joblib
import datetime
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import ValueWarning
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import joblib
from tqdm import tqdm

class UseCase03Model(object):
  def __init__(self, use_store=False, use_department=True):
      if not use_store and not use_department:
          raise ValueError(f'use_store = {{use_store}}, use_department = {{use_department}}: at least one must be True')

      self._use_store = use_store
      self._use_department = use_department
      self._models = {{}}
      self._min = {{}}
      self._max = {{}}

  def _get_key(self, store, department):
      if self._use_store and self._use_department:
          key = (store, department)
      elif self._use_store:
          key = store
      else:
          key = department

      return key

  def store_model(self, store: int, department: int, model, ts_min, ts_max):
      key = self._get_key(store, department)
      self._models[key] = model
      self._min[key] = ts_min
      self._max[key] = ts_max

  def get_model(self, store: int, department: int):
      key = self._get_key(store, department)
      model = self._models[key]
      ts_min = self._min[key]
      ts_max = self._max[key]
      return model, ts_min, ts_max

def pyinitial():
    global uc03_model
    model_file = '{1}/tpcxai_datasets/{0}/model/uc03/uc03.python.model'
    uc03_model = joblib.load(model_file)

def pyfun(*args):
    forecasts = []
    data = pd.DataFrame({{
        'store': args[0],
        'department': args[1]
    }})
    for index, row in data.iterrows():
        store = row.store
        dept = row.department
        periods = 52
        try:
            current_model, ts_min, ts_max = uc03_model.get_model(store, dept)
        except KeyError:
            continue
        # disable warnings that non-date index is returned from forecast
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=ValueWarning)
            forecast = current_model.forecast(periods)
            forecast = np.clip(forecast, a_min=0.0, a_max=None)  # replace negative forecasts
        start = pd.date_range(ts_max, periods=2)[1]
        forecast_idx = pd.date_range(start, periods=periods, freq='W-FRI')
        forecasts.append(
            str({{'store': store, 'department': dept, 'date': forecast_idx, 'weekly_sales': forecast}})
        )
    return np.array(forecasts)
"}};'''

tpcx_ai_uc03_uninit = '''CREATE PYTHON_UDF tpcx_ai_uc03_{0}_uninit(store INTEGER, department STRING) RETURNS STRING 
{{"
import joblib
import datetime
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import ValueWarning
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import joblib
from tqdm import tqdm

class UseCase03Model(object):
  def __init__(self, use_store=False, use_department=True):
      if not use_store and not use_department:
          raise ValueError(f'use_store = {{use_store}}, use_department = {{use_department}}: at least one must be True')

      self._use_store = use_store
      self._use_department = use_department
      self._models = {{}}
      self._min = {{}}
      self._max = {{}}

  def _get_key(self, store, department):
      if self._use_store and self._use_department:
          key = (store, department)
      elif self._use_store:
          key = store
      else:
          key = department

      return key

  def store_model(self, store: int, department: int, model, ts_min, ts_max):
      key = self._get_key(store, department)
      self._models[key] = model
      self._min[key] = ts_min
      self._max[key] = ts_max

  def get_model(self, store: int, department: int):
      key = self._get_key(store, department)
      model = self._models[key]
      ts_min = self._min[key]
      ts_max = self._max[key]
      return model, ts_min, ts_max

def pyinitial():
    pass

def pyfun(*args):
    model_file = '{1}/tpcxai_datasets/{0}/model/uc03/uc03.python.model'
    uc03_model = joblib.load(model_file)
    forecasts = []
    data = pd.DataFrame({{
        'store': args[0],
        'department': args[1]
    }})
    for index, row in data.iterrows():
        store = row.store
        dept = row.department
        periods = 52
        try:
            current_model, ts_min, ts_max = uc03_model.get_model(store, dept)
        except KeyError:
            continue
        # disable warnings that non-date index is returned from forecast
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=ValueWarning)
            forecast = current_model.forecast(periods)
            forecast = np.clip(forecast, a_min=0.0, a_max=None)  # replace negative forecasts
        start = pd.date_range(ts_max, periods=2)[1]
        forecast_idx = pd.date_range(start, periods=periods, freq='W-FRI')
        forecasts.append(
            str({{'store': store, 'department': dept, 'date': forecast_idx, 'weekly_sales': forecast}})
        )
    return np.array(forecasts)
"}};'''

tpcx_ai_uc04 = '''CREATE PYTHON_UDF tpcx_ai_uc04_{0}_init(text_ STRING) RETURNS INTEGER 
{{"
import joblib
import numpy as np
import pandas as pd
import time
from duckdb.typing import BIGINT, DOUBLE, FLOAT, VARCHAR
from sklearn import feature_extraction, naive_bayes
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
def pyinitial():
    start = time.perf_counter()
    global uc04_model
    model_file = '{1}/tpcxai_datasets/{0}/model/uc04/uc04.python.model'
    uc04_model = joblib.load(model_file)
    stop = time.perf_counter()
    with open('/home/test/experiments/oceanbase/opt/both_pps.log', 'a+') as log:
        log.write(str(stop-start)+' ')

def pyfun(*args):
    data = pd.DataFrame({{
        'text': args[0]
    }})
    return uc04_model.predict(data['text'])
"}};'''

tpcx_ai_uc04_uninit = '''CREATE PYTHON_UDF tpcx_ai_uc04_{0}_uninit(text_ STRING) RETURNS INTEGER 
{{"
import joblib
import time
import numpy as np
import pandas as pd
from duckdb.typing import BIGINT, DOUBLE, FLOAT, VARCHAR
from sklearn import feature_extraction, naive_bayes
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
def pyinitial():
    pass

def pyfun(*args):
    start = time.perf_counter()
    model_file = '{1}/tpcxai_datasets/{0}/model/uc04/uc04.python.model'
    uc04_model = joblib.load(model_file)
    stop = time.perf_counter()
    with open('/home/test/experiments/oceanbase/opt/both_pps.log', 'a+') as log:
        log.write(str(stop-start)+' ')
    data = pd.DataFrame({{
        'text': args[0]
    }})
    return uc04_model.predict(data['text'])
"}};'''

tpcx_ai_uc06 = '''CREATE PYTHON_UDF tpcx_ai_uc06_{0}_init(smart_5_raw REAL, smart_10_raw REAL, smart_184_raw REAL, smart_187_raw REAL, smart_188_raw REAL, smart_197_raw REAL, smart_198_raw REAL) RETURNS INTEGER 
{{"
import joblib
import numpy as np
import pandas as pd
from duckdb.typing import BIGINT, DOUBLE, FLOAT, VARCHAR
from imblearn.over_sampling import ADASYN
from imblearn.pipeline import Pipeline as Imb_Pipeline
import joblib
import time
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
def pyinitial():
    start = time.perf_counter()
    global uc06_model
    model_file = '{1}/tpcxai_datasets/{0}/model/uc06/uc06.python.model'
    uc06_model = joblib.load(model_file)
    stop = time.perf_counter()
    with open('/home/test/experiments/oceanbase/opt/both_pps.log', 'a+') as log:
        log.write(str(stop-start)+' ')

def pyfun(*args):
    data = pd.DataFrame({{
        'smart_5_raw': args[0],
        'smart_10_raw': args[1],
        'smart_184_raw': args[2],
        'smart_187_raw': args[3],
        'smart_188_raw': args[4],
        'smart_197_raw': args[5],
        'smart_198_raw': args[6]
    }})
    return uc06_model.predict(data)
"}};'''

tpcx_ai_uc06_uninit = '''CREATE PYTHON_UDF tpcx_ai_uc06_{0}_uninit(smart_5_raw REAL, smart_10_raw REAL, smart_184_raw REAL, smart_187_raw REAL, smart_188_raw REAL, smart_197_raw REAL, smart_198_raw REAL) RETURNS INTEGER 
{{"
import joblib
import numpy as np
import pandas as pd
from duckdb.typing import BIGINT, DOUBLE, FLOAT, VARCHAR
from imblearn.over_sampling import ADASYN
from imblearn.pipeline import Pipeline as Imb_Pipeline
import joblib
import time
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
def pyinitial():
    pass

def pyfun(*args):
    start = time.perf_counter()
    model_file = '{1}/tpcxai_datasets/{0}/model/uc06/uc06.python.model'
    uc06_model = joblib.load(model_file)
    stop = time.perf_counter()
    with open('/home/test/experiments/oceanbase/opt/both_pps.log', 'a+') as log:
        log.write(str(stop-start)+' ')
    data = pd.DataFrame({{
        'smart_5_raw': args[0],
        'smart_10_raw': args[1],
        'smart_184_raw': args[2],
        'smart_187_raw': args[3],
        'smart_188_raw': args[4],
        'smart_197_raw': args[5],
        'smart_198_raw': args[6]
    }})
    return uc06_model.predict(data)
"}};'''

tpcx_ai_uc07 = '''CREATE PYTHON_UDF tpcx_ai_uc07_{0}_init(user_id INTEGER, item_id INTEGER) RETURNS REAL 
{{"
import joblib
import numpy as np
import pandas as pd
import joblib
import time
from surprise import SVD
from surprise import Dataset
from surprise.reader import Reader
def pyinitial():
    start = time.perf_counter()
    global uc07_model
    model_file = '{1}/tpcxai_datasets/{0}/model/uc07/uc07.python.model'
    uc07_model = joblib.load(model_file)
    stop = time.perf_counter()
    with open('/home/test/experiments/oceanbase/opt/both_pps.log', 'a+') as log:
        log.write(str(stop-start)+' ')

def pyfun(*args):
    user_id = args[0]
    item_id = args[1]
    ratings = []
    for i in range(len(user_id)):
        rating = uc07_model.predict(user_id[i], item_id[i]).est
        ratings.append(rating)
    return np.array(ratings)
"}};'''

tpcx_ai_uc07_uninit = '''CREATE PYTHON_UDF tpcx_ai_uc07_{0}_uninit(user_id INTEGER, item_id INTEGER) RETURNS REAL 
{{"
import joblib
import numpy as np
import pandas as pd
import joblib
import time
from surprise import SVD
from surprise import Dataset
from surprise.reader import Reader
def pyinitial():
    pass

def pyfun(*args):
    start = time.perf_counter()
    model_file = '{1}/tpcxai_datasets/{0}/model/uc07/uc07.python.model'
    uc07_model = joblib.load(model_file)
    user_id = args[0]
    item_id = args[1]
    ratings = []
    stop = time.perf_counter()
    for i in range(len(user_id)):
        rating = uc07_model.predict(user_id[i], item_id[i]).est
        ratings.append(rating)
    return np.array(ratings)
"}};'''

tpcx_ai_uc08 = '''CREATE PYTHON_UDF tpcx_ai_uc08_{0}_init(scan_count INTEGER, scan_count_abs INTEGER, Monday INTEGER, Tuesday INTEGER, Wednesday INTEGER, Thursday INTEGER, Friday INTEGER, Saturday INTEGER, Sunday INTEGER, dep0 INTEGER, dep1 INTEGER, dep2 INTEGER, dep3 INTEGER, dep4 INTEGER, dep5 INTEGER, dep6 INTEGER, dep7 INTEGER, dep8 INTEGER, dep9 INTEGER, dep10 INTEGER, dep11 INTEGER, dep12 INTEGER, dep13 INTEGER, dep14 INTEGER, dep15 INTEGER, dep16 INTEGER, dep17 INTEGER, dep18 INTEGER, dep19 INTEGER, dep20 INTEGER, dep21 INTEGER, dep22 INTEGER, dep23 INTEGER, dep24 INTEGER, dep25 INTEGER, dep26 INTEGER, dep27 INTEGER, dep28 INTEGER, dep29 INTEGER, dep30 INTEGER, dep31 INTEGER, dep32 INTEGER, dep33 INTEGER, dep34 INTEGER, dep35 INTEGER, dep36 INTEGER, dep37 INTEGER, dep38 INTEGER, dep39 INTEGER, dep40 INTEGER, dep41 INTEGER, dep42 INTEGER, dep43 INTEGER, dep44 INTEGER, dep45 INTEGER, dep46 INTEGER, dep47 INTEGER, dep48 INTEGER, dep49 INTEGER, dep50 INTEGER, dep51 INTEGER, dep52 INTEGER, dep53 INTEGER, dep54 INTEGER, dep55 INTEGER, dep56 INTEGER, dep57 INTEGER, dep58 INTEGER, dep59 INTEGER, dep60 INTEGER, dep61 INTEGER, dep62 INTEGER, dep63 INTEGER, dep64 INTEGER, dep65 INTEGER, dep66 INTEGER, dep67 INTEGER) RETURNS INTEGER 
{{"
import joblib
import numpy as np
import pandas as pd
from duckdb.typing import BIGINT, DOUBLE, FLOAT, VARCHAR,HUGEINT
from scipy.sparse import csr_matrix
from xgboost.sklearn import XGBClassifier

label_range = [3, 4, 5, 6, 7, 8, 9, 12, 15, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 999]
sorted_labels = sorted(label_range, key=str)

def decode_label(label):
    return sorted_labels[label]

def pyinitial():
    global uc08_model
    model_file = '{1}/tpcxai_datasets/{0}/model/uc08/uc08.python.model'
    uc08_model = joblib.load(model_file)

def pyfun(*args):
    data = pd.DataFrame({{'scan_count': args[0], 'scan_count_abs': args[1], 'Monday': args[2], 'Tuesday': args[3], 'Wednesday': args[4], 'Thursday': args[5], 'Friday': args[6], 'Saturday': args[7], 'Sunday': args[8], 'dep0': args[9], 'dep1': args[10], 'dep2': args[11], 'dep3': args[12], 'dep4': args[13], 'dep5': args[14], 'dep6': args[15], 'dep7': args[16], 'dep8': args[17], 'dep9': args[18], 'dep10': args[19], 'dep11': args[20], 'dep12': args[21], 'dep13': args[22], 'dep14': args[23], 'dep15': args[24], 'dep16': args[25], 'dep17': args[26], 'dep18': args[27], 'dep19': args[28], 'dep20': args[29], 'dep21': args[30], 'dep22': args[31], 'dep23': args[32], 'dep24': args[33], 'dep25': args[34], 'dep26': args[35], 'dep27': args[36], 'dep28': args[37], 'dep29': args[38], 'dep30': args[39], 'dep31': args[40], 'dep32': args[41], 'dep33': args[42], 'dep34': args[43], 'dep35': args[44], 'dep36': args[45], 'dep37': args[46], 'dep38': args[47], 'dep39': args[48], 'dep40': args[49], 'dep41': args[50], 'dep42': args[51], 'dep43': args[52], 'dep44': args[53], 'dep45': args[54], 'dep46': args[55], 'dep47': args[56], 'dep48': args[57], 'dep49': args[58], 'dep50': args[59], 'dep51': args[60], 'dep52': args[61], 'dep53': args[62], 'dep54': args[63], 'dep55': args[64], 'dep56': args[65], 'dep57': args[66], 'dep58': args[67], 'dep59': args[68], 'dep60': args[69], 'dep61': args[70], 'dep62': args[71], 'dep63': args[72], 'dep64': args[73], 'dep65': args[74], 'dep66': args[75], 'dep67': args[76]}})

    sparse_data = csr_matrix(data)
    predictions = uc08_model.predict(sparse_data)
    dec_fun = np.vectorize(decode_label)

    return dec_fun(predictions)
"}};'''

tpcx_ai_uc10 = '''CREATE PYTHON_UDF tpcx_ai_uc10_{0}_init(business_hour_norm REAL, amount_norm REAL) RETURNS INTEGER 
{{"
import joblib
import numpy as np
import time
import pandas as pd
from sklearn.linear_model import LogisticRegression
def pyinitial():
    start = time.perf_counter()
    global uc10_model
    model_file = '{1}/tpcxai_datasets/{0}/model/uc10/uc10.python.model'
    uc10_model = joblib.load(model_file)
    stop = time.perf_counter()
    with open('/home/test/experiments/oceanbase/opt/both_pps.log', 'a+') as log:
        log.write(str(stop-start)+' ')
    
def pyfun(*args):
    data = pd.DataFrame({{
        'business_hour_norm': args[0],
        'amount_norm': args[1]
    }})
    return uc10_model.predict(data)
"}};'''

tpcx_ai_uc10_uninit = '''CREATE PYTHON_UDF tpcx_ai_uc10_{0}_uninit(business_hour_norm REAL, amount_norm REAL) RETURNS INTEGER 
{{"
import joblib
import numpy as np
import pandas as pd
import time
from sklearn.linear_model import LogisticRegression
def pyinitial():
    pass

def pyfun(*args):
    start = time.perf_counter()
    model_file = '{1}/tpcxai_datasets/{0}/model/uc10/uc10.python.model'
    uc10_model = joblib.load(model_file)
    stop = time.perf_counter()
    data = pd.DataFrame({{
        'business_hour_norm': args[0],
        'amount_norm': args[1]
    }})
    return uc10_model.predict(data)
"}};'''

test_efficiency = ''

test_torch = '''CREATE PYTHON_UDF test_torch(input1 INTEGER) RETURNS INTEGER 
{"
import numpy as np
import torch
import time
def pyinitial():
    pass
def pyfun(*args):
    start = time.process_time()
    m1 = np.random.randint(0,10,(10,10))
    m2 = np.random.randint(0,10,(10,10))
    tensor1 = torch.tensor(m1, dtype=torch.float32)
    tensor2 = torch.tensor(m2, dtype=torch.float32)

    # result = torch.matmul(tensor1, tensor2)

    finish = time.process_time()
    return args[0]
"};'''

test_tensorflow = '''CREATE PYTHON_UDF test_tensorflow(input1 INTEGER) RETURNS INTEGER
{"
import numpy as np
import tensorflow as tf
def pyinitial():
	pass
def pyfun(*args):
	m1 = tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.float32)
	m2 = tf.constant([[1, 1, 1], [2, 2, 2]], dtype=tf.float32)
	result = tf.multiply(m1, m2)
	return args[0]
"};
'''

drop_udf = '''DROP PYTHON_UDF {};'''

def set_query_raven(test_sql):
    # create python udf
	#test_sql.append(expedia_onnx_dt_init.format(prefix))
	#test_sql.append(expedia_onnx_dt_uninit.format(prefix))
    #test_sql.append(expedia_sklearn_dt_init.format(prefix))
    #test_sql.append(expedia_sklearn_dt_uninit.format(prefix))
    #test_sql.append(flights_sklearn_rf_init.format(prefix))
    #test_sql.append(flights_sklearn_rf_uninit.format(prefix))
    #test_sql.append(hospital_sklearn_lr_init.format(prefix))
    #test_sql.append(hospital_sklearn_lr_uninit.format(prefix))
    #test_sql.append(hospital_pytorch_mlp_init.format(prefix))
    #test_sql.append(hospital_pytorch_mlp_uninit.format(prefix))
    #test_sql.append(credit_card_lightgbm_init.format(prefix))
    #test_sql.append(credit_card_lightgbm_uninit.format(prefix))
	#test_sql.append(credit_card_xgboost_init.format(prefix))
	
    # test_sql.append(expedia_onnx_dt_uninit.format(prefix)) # Q1
    # test_sql.append(expedia_sklearn_dt_uninit.format(prefix)) # Q2
    # test_sql.append(flights_sklearn_rf_uninit.format(prefix)) # Q3
    # test_sql.append(hospital_pytorch_mlp_uninit.format(prefix)) # Q4
    # test_sql.append(credit_card_lightgbm_uninit.format(prefix)) # Q5
    # test_sql.append(credit_card_xgboost_uninit.format(prefix)) # Q6

    # test_sql.append(expedia_onnx_dt_uninit.format(prefix)) # Q1
    # test_sql.append(expedia_sklearn_dt_uninit.format(prefix)) # Q2
    # test_sql.append(flights_sklearn_rf_uninit.format(prefix)) # Q3
    # test_sql.append(hospital_pytorch_mlp_uninit.format(prefix)) # Q4
    # test_sql.append(credit_card_lightgbm_uninit.format(prefix)) # Q5
    # test_sql.append(credit_card_xgboost_uninit.format(prefix)) # Q6

    test_sql.append(expedia_sklearn_dt_uninit.format(prefix)) # Q2
    test_sql.append(expedia_sklearn_dt_init.format(prefix)) # Q2
    test_sql.append(hospital_pytorch_mlp_uninit.format(prefix)) # Q4
    test_sql.append(hospital_pytorch_mlp_init.format(prefix)) # Q4

def set_udf_trees(test_sql):
    for elem in [10,50,100,500,1000]:
        test_sql.append(flights_sklearn_rf_uninit_trees.format(prefix, elem)) # Q3
        test_sql.append(credit_card_lightgbm_uninit_trees.format(prefix, elem)) # Q5
        test_sql.append(credit_card_xgboost_uninit_trees.format(prefix, elem)) # Q6

def set_drop_udf_trees(sql):
    for elem in [10,50,100,500,1000]:
        sql.append(drop_udf.format(f"flights_sklearn_rf_uninit_trees{elem}")) # Q3
        sql.append(drop_udf.format(f"creditcard_lightgbm_gb_uninit_trees{elem}")) # Q5
        sql.append(drop_udf.format(f"creditcard_xgboost_gb_uninit_trees{elem}")) # Q6

def set_query_tpcx_ai(test_sql):
    '''
    test_sql.append(tpcx_ai_uc01.format("sf10", prefix))
    test_sql.append(tpcx_ai_uc03.format("sf10", prefix))
    test_sql.append(tpcx_ai_uc04.format("sf10", prefix))
    test_sql.append(tpcx_ai_uc06.format("sf10", prefix))
    test_sql.append(tpcx_ai_uc07.format("sf10", prefix))
    test_sql.append(tpcx_ai_uc08.format("sf10", prefix))
    test_sql.append(tpcx_ai_uc10.format("sf10", prefix))
    '''
    # create python udf
	#test_sql.append(tpcx_ai_uc01_uninit.format("sf10", prefix))
	#test_sql.append(tpcx_ai_uc03_uninit.format("sf10", prefix))
	#test_sql.append(tpcx_ai_uc04_uninit.format("sf10", prefix))
	#test_sql.append(tpcx_ai_uc06_uninit.format("sf10", prefix))
	#test_sql.append(tpcx_ai_uc07_uninit.format("sf10", prefix))
	#test_sql.append(tpcx_ai_uc10_uninit.format("sf10", prefix))
      
    # test_sql.append(tpcx_ai_uc01.format("sf10", prefix))
    # test_sql.append(tpcx_ai_uc03.format("sf10", prefix))
    # test_sql.append(tpcx_ai_uc04.format("sf10", prefix))
    # test_sql.append(tpcx_ai_uc06.format("sf10", prefix))
    # test_sql.append(tpcx_ai_uc07.format("sf10", prefix))
    # test_sql.append(tpcx_ai_uc10.format("sf10", prefix))

    for elem in [20, 40 ,60, 80, 100]:
        test_sql.append(tpcx_ai_uc03_uninit.format(f"sf{elem}", prefix))
        test_sql.append(tpcx_ai_uc07_uninit.format(f"sf{elem}", prefix))
        test_sql.append(tpcx_ai_uc10_uninit.format(f"sf{elem}", prefix))

    # test_sql.append(tpcx_ai_uc06.format("sf10", prefix))
    # test_sql.append(tpcx_ai_uc06_uninit.format("sf10", prefix))
    # test_sql.append(tpcx_ai_uc07_uninit.format("sf10", prefix))
    # test_sql.append(tpcx_ai_uc07.format("sf10", prefix))
	
def set_query_drop_udf(sql):
	#sql.append(drop_udf.format("tpcx_ai_uc01_sf10"))
	#sql.append(drop_udf.format("tpcx_ai_uc03_sf10"))
	##sql.append(drop_udf.format("tpcx_ai_uc04_sf10"))
	#sql.append(drop_udf.format("tpcx_ai_uc06_sf10"))
	#sql.append(drop_udf.format("tpcx_ai_uc07_sf10"))
	#sql.append(drop_udf.format("tpcx_ai_uc08_sf10"))
	#sql.append(drop_udf.format("tpcx_ai_uc10_sf10"))


    #sql.append(drop_udf.format("flights_sklearn_rf_init"))
    #sql.append(drop_udf.format("hospital_sklearn_lr_uninit"))
    #sql.append(drop_udf.format("credit_card_lightgbm_uninit"))
	#sql.append(drop_udf.format("creditcard_xgboost_gb_init"))
    
    #sql.append(drop_udf.format("test_torch"))
	#sql.append(drop_udf.format("test_tensorflow"))
    #sql.append(drop_udf.format("hospital_pytorch_mlp_init"))
	
    sql.append(drop_udf.format("expedia_onnx_dt_uninit")) # Q1
    sql.append(drop_udf.format("expedia_sklearn_dt_uninit")) # Q2
    sql.append(drop_udf.format("flights_sklearn_rf_uninit")) # Q3
    sql.append(drop_udf.format("hospital_pytorch_mlp_uninit")) # Q4
    sql.append(drop_udf.format("creditcard_lightgbm_gb_uninit")) # Q5
    sql.append(drop_udf.format("creditcard_xgboost_gb_uninit")) # Q6

def set_query_drop_udf_cache_overhead(sql):
    sql.append(drop_udf.format("expedia_sklearn_dt_uninit")) # Q2
    sql.append(drop_udf.format("expedia_sklearn_dt_init")) # Q2
    sql.append(drop_udf.format("hospital_pytorch_mlp_uninit")) # Q4
    sql.append(drop_udf.format("hospital_pytorch_mlp_init")) # Q4

    sql.append(drop_udf.format("tpcx_ai_uc06_sf10_uninit")) # Q10
    sql.append(drop_udf.format("tpcx_ai_uc06_sf10_init")) # Q10
    sql.append(drop_udf.format("tpcx_ai_uc07_sf10_uninit")) # Q11
    sql.append(drop_udf.format("tpcx_ai_uc07_sf10_init")) # Q11

def set_test_udf(sql):
    #sql.append(test_torch)
    sql.append(test_tensorflow)

'''
# raven connection
conn = pymysql.connect(host="127.0.0.1", port=2881, user="root", passwd="ulAFVBT0D4GDfMizqJXJ", db="raven")
cur = conn.cursor()

try:
	test_sql = []
	time_stat = 0
	cur.execute(trace_on) # open trace
	set_query_raven(test_sql)
	for i in test_sql:
		time_consuming = run_sql(cur, i)
		df = pd.DataFrame({'query': [i], 'execute': [time_consuming], 'time': [time.asctime()]})
		df.to_csv(output_path, index=True, mode='a', header=None)

finally:
  cur.close()
  conn.close()
'''

# tpcx_ai connection
conn = pymysql.connect(host="127.0.0.1", port=2881, user="root", passwd="ulAFVBT0D4GDfMizqJXJ", db="tpcx_ai")
cur = conn.cursor()

try:
    test_sql = []
    time_stat = 0
    cur.execute(trace_on) # open trace
    # set_query_drop_udf(test_sql)
    # set_query_drop_udf_cache_overhead(test_sql)
    # set_query_raven(test_sql)
    set_query_tpcx_ai(test_sql)
    # set_drop_udf_trees(test_sql)
    # set_udf_trees(test_sql)
    #set_test_udf(test_sql)
    for i in test_sql:
        time_consuming = run_sql(cur, i)
        df = pd.DataFrame({'query': [i], 'execute': [time_consuming], 'time': [time.asctime()]})
        df.to_csv(output_path, index=True, mode='a', header=None)

finally:
    cur.close()
    conn.close()