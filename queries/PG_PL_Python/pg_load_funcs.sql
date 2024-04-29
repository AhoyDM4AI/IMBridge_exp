CREATE FUNCTION uc01(return_ratio REAL, frequency REAL)
RETURNS INTEGER
AS $$
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

name = "uc01"
model_file_name = f"/home/tpcxai_datasets/model/{name}/{name}.python.model"

model = joblib.load(model_file_name)
feat = pd.DataFrame({
    'return_ratio': np.array([return_ratio]),
    'frequency': np.array([frequency])
})

return model.predict(feat)[0]
$$ LANGUAGE plpython3u;

CREATE FUNCTION uc03(store text, department text)
RETURNS text
AS $$
import warnings
import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import ValueWarning
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import joblib

name = "uc03"
model_file_name = f"/home/tpcxai_datasets/model/{name}/{name}.python.model"

main = __import__('__main__')

class UseCase03Model(object):

    def __init__(self, use_store=False, use_department=True):
        if not use_store and not use_department:
            raise ValueError(f"use_store = {use_store}, use_department = {use_department}: at least one must be True")

        self._use_store = use_store
        self._use_department = use_department
        self._models = {}
        self._min = {}
        self._max = {}

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
main.UseCase03Model = UseCase03Model
models = joblib.load(model_file_name)

forecasts = []
periods = 52
try:
    current_model, ts_min, ts_max = models.get_model(store, department)
except KeyError:
    return str(type(store))
# disable warnings that non-date index is returned from forecast
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=ValueWarning)
    forecast = current_model.forecast(periods)
    forecast = np.clip(forecast, a_min=0.0, a_max=None)  # replace negative forecasts
start = pd.date_range(ts_max, periods=2)[1]
forecast_idx = pd.date_range(start, periods=periods, freq='W-FRI')
forecasts.append(
    str({'store': store, 'department': department, 'date': forecast_idx, 'weekly_sales': forecast})
)

return forecasts[0]
$$ LANGUAGE plpython3u;


CREATE FUNCTION uc04(txt text)
RETURNS INTEGER
AS $$
import joblib
import numpy as np
import pandas as pd
from sklearn import feature_extraction, naive_bayes
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer

name = "uc04"
model_file_name = f"/home/tpcxai_datasets/model/{name}/{name}.python.model"

model = joblib.load(model_file_name)
data = pd.DataFrame({
    "text": np.array([txt])
})
return model.predict(data["text"])[0]

$$ LANGUAGE plpython3u;


CREATE FUNCTION uc06(smart_5_raw REAL,smart_10_raw REAL,smart_184_raw REAL,
                     smart_187_raw REAL,smart_188_raw REAL,smart_197_raw REAL,smart_198_raw REAL)
RETURNS INTEGER
AS $$
import numpy as np
import pandas as pd
from imblearn.over_sampling import ADASYN
from imblearn.pipeline import Pipeline as Imb_Pipeline
import joblib
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

name = "uc06"
model_file_name = f"/home/tpcxai_datasets/model/{name}/{name}.python.model"

model = joblib.load(model_file_name)

data = pd.DataFrame({
    'smart_5_raw': np.array([smart_5_raw]),
    'smart_10_raw': np.array([smart_10_raw]),
    'smart_184_raw': np.array([smart_184_raw]),
    'smart_187_raw': np.array([smart_187_raw]),
    'smart_188_raw': np.array([smart_188_raw]),
    'smart_197_raw': np.array([smart_197_raw]),
    'smart_198_raw': np.array([smart_198_raw])
})
return model.predict(data)[0]

$$ LANGUAGE plpython3u;

CREATE FUNCTION uc07(user_id INTEGER, item_id INTEGER)
RETURNS REAL
AS $$
import numpy as np
import pandas as pd
import joblib
from surprise import SVD
from surprise import Dataset
from surprise.reader import Reader

name = "uc07"
model_file_name = f"/home/tpcxai_datasets/model/{name}/{name}.python.model"

model = joblib.load(model_file_name)
rating = model.predict(user_id, item_id).est
return rating
$$ LANGUAGE plpython3u;

CREATE FUNCTION uc08(scan_count INTEGER, scan_count_abs INTEGER, Monday INTEGER, Tuesday INTEGER, Wednesday INTEGER,
                     Thursday INTEGER, Friday INTEGER, Saturday INTEGER, Sunday INTEGER, dep0 INTEGER, dep1 INTEGER,
                     dep2 INTEGER, dep3 INTEGER, dep4 INTEGER, dep5 INTEGER, dep6 INTEGER, dep7 INTEGER, dep8 INTEGER,
                     dep9 INTEGER, dep10 INTEGER, dep11 INTEGER, dep12 INTEGER, dep13 INTEGER, dep14 INTEGER, dep15 INTEGER,
                     dep16 INTEGER, dep17 INTEGER, dep18 INTEGER, dep19 INTEGER, dep20 INTEGER, dep21 INTEGER, dep22 INTEGER,
                     dep23 INTEGER, dep24 INTEGER, dep25 INTEGER, dep26 INTEGER, dep27 INTEGER, dep28 INTEGER, dep29 INTEGER,
                     dep30 INTEGER, dep31 INTEGER, dep32 INTEGER, dep33 INTEGER, dep34 INTEGER, dep35 INTEGER, dep36 INTEGER,
                     dep37 INTEGER, dep38 INTEGER, dep39 INTEGER, dep40 INTEGER, dep41 INTEGER, dep42 INTEGER, dep43 INTEGER,
                     dep44 INTEGER, dep45 INTEGER, dep46 INTEGER, dep47 INTEGER, dep48 INTEGER, dep49 INTEGER, dep50 INTEGER,
                     dep51 INTEGER, dep52 INTEGER, dep53 INTEGER, dep54 INTEGER, dep55 INTEGER, dep56 INTEGER, dep57 INTEGER,
                     dep58 INTEGER, dep59 INTEGER, dep60 INTEGER, dep61 INTEGER, dep62 INTEGER, dep63 INTEGER, dep64 INTEGER,
                     dep65 INTEGER, dep66 INTEGER, dep67 INTEGER)
RETURNS INTEGER
AS $$
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from xgboost.sklearn import XGBClassifier

name = "uc08"
model_file_name = f"/home/tpcxai_datasets/model/{name}/{name}.python.model"

model = joblib.load(model_file_name)

label_range = [3, 4, 5, 6, 7, 8, 9, 12, 15, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
               37, 38, 39, 40, 41, 42, 43, 44, 999]
sorted_labels = sorted(label_range, key=str)


def decode_label(label):
    return sorted_labels[label]

sparse_data = csr_matrix(np.array(args))
predictions = model.predict(sparse_data)
dec_fun = np.vectorize(decode_label)

return dec_fun(predictions)[0]

$$ LANGUAGE plpython3u;


CREATE FUNCTION uc10(business_hour_norm REAL, amount_norm REAL)
RETURNS REAL
AS $$
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

name = "uc10"
model_file_name = f"/home/tpcxai_datasets/model/{name}/{name}.python.model"

model = joblib.load(model_file_name)

data = pd.DataFrame({
    'business_hour_norm': np.array([business_hour_norm]),
    'amount_norm': np.array([amount_norm])
})
return model.predict(data)[0]

$$ LANGUAGE plpython3u;

CREATE FUNCTION pf1(prop_location_score1 REAL, prop_location_score2 REAL,
                    prop_log_historical_price REAL, price_usd REAL, orig_destination_distance REAL,
                    prop_review_score REAL, avg_bookings_usd REAL, stdev_bookings_usd REAL, position_ VARCHAR,
                    prop_country_id VARCHAR, prop_starrating INTEGER, prop_brand_bool INTEGER,
                    count_clicks INTEGER, count_bookings INTEGER, year_ VARCHAR, month_ VARCHAR,
                    weekofyear_ VARCHAR, time_ VARCHAR, site_id VARCHAR, visitor_location_country_id VARCHAR,
                    srch_destination_id VARCHAR, srch_length_of_stay INTEGER, srch_booking_window INTEGER,
                    srch_adults_count INTEGER, srch_children_count INTEGER, srch_room_count INTEGER,
                    srch_saturday_night_bool INTEGER, random_bool INTEGER)
RETURNS INTEGER
AS $$
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import time

name = "pf1"


start = time.perf_counter()
scaler_path = '/home/public_datasets/Expedia/expedia_standard_scale_model.pkl'
enc_path = '/home/public_datasets/Expedia/expedia_one_hot_encoder.pkl'
model_path = '/home/public_datasets/Expedia/expedia_dt_model.pkl'
with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)
with open(enc_path, 'rb') as f:
    enc = pickle.load(f)
with open(model_path, 'rb') as f:
    model = pickle.load(f)
stop = time.perf_counter()

with open(f"/home/pg_init_{name}.log", 'a+') as f:
    f.write(f"{(stop-start)*1000}\n")

numerical = [args[0:8]]
categorical = [args[8:]]

X = np.hstack((scaler.transform(numerical), enc.transform(categorical).toarray()))
return model.predict(X)[0]
$$ LANGUAGE plpython3u;

CREATE FUNCTION pf2(prop_location_score1 REAL, prop_location_score2 REAL,
                    prop_log_historical_price REAL, price_usd REAL, orig_destination_distance REAL,
                    prop_review_score REAL, avg_bookings_usd REAL, stdev_bookings_usd REAL, position_ VARCHAR,
                    prop_country_id VARCHAR, prop_starrating INTEGER, prop_brand_bool INTEGER,
                    count_clicks INTEGER, count_bookings INTEGER, year_ VARCHAR, month_ VARCHAR,
                    weekofyear_ VARCHAR, time_ VARCHAR, site_id VARCHAR, visitor_location_country_id VARCHAR,
                    srch_destination_id VARCHAR, srch_length_of_stay INTEGER, srch_booking_window INTEGER,
                    srch_adults_count INTEGER, srch_children_count INTEGER, srch_room_count INTEGER,
                    srch_saturday_night_bool INTEGER, random_bool INTEGER)
RETURNS INTEGER
AS $$
import numpy as np
import pandas as pd
import time
import onnxruntime as ort
import time

name = "pf2"

start = time.perf_counter()
onnx_path = '/home/public_datasets/Expedia/expedia_dt_pipeline.onnx'
ortconfig = ort.SessionOptions()
expedia_onnx_session = ort.InferenceSession(onnx_path, sess_options=ortconfig)
expedia_label = expedia_onnx_session.get_outputs()[0]
numerical_columns = ['prop_location_score1', 'prop_location_score2', 'prop_log_historical_price', 'price_usd',
                     'orig_destination_distance', 'prop_review_score', 'avg_bookings_usd', 'stdev_bookings_usd']
categorical_columns = ['position', 'prop_country_id', 'prop_starrating', 'prop_brand_bool', 'count_clicks',
                       'count_bookings', 'year', 'month', 'weekofyear', 'time', 'site_id',
                       'visitor_location_country_id', 'srch_destination_id', 'srch_length_of_stay',
                       'srch_booking_window', 'srch_adults_count', 'srch_children_count', 'srch_room_count',
                       'srch_saturday_night_bool', 'random_bool']
expedia_input_columns = numerical_columns + categorical_columns
expedia_type_map = {
    int: np.int64,
    bool: np.int64,
    float: np.float32,
    str: str
}
stop = time.perf_counter()
with open(f"/home/pg_init_{name}.log", 'a+') as f:
    f.write(f"{(stop-start)*1000}\n")

infer_batch = {
    elem: np.array([args[i]]).astype(expedia_type_map[type(args[i])]).reshape((-1, 1))
    for i, elem in enumerate(expedia_input_columns)
}
outputs = expedia_onnx_session.run([expedia_label.name], infer_batch)

return outputs[0][0]

$$ LANGUAGE plpython3u;

CREATE FUNCTION pf2(prop_location_score1 REAL, prop_location_score2 REAL,
                    prop_log_historical_price REAL, price_usd REAL, orig_destination_distance REAL,
                    prop_review_score REAL, avg_bookings_usd REAL, stdev_bookings_usd REAL, position_ VARCHAR,
                    prop_country_id VARCHAR, prop_starrating INTEGER, prop_brand_bool INTEGER,
                    count_clicks INTEGER, count_bookings INTEGER, year_ VARCHAR, month_ VARCHAR,
                    weekofyear_ VARCHAR, time_ VARCHAR, site_id VARCHAR, visitor_location_country_id VARCHAR,
                    srch_destination_id VARCHAR, srch_length_of_stay INTEGER, srch_booking_window INTEGER,
                    srch_adults_count INTEGER, srch_children_count INTEGER, srch_room_count INTEGER,
                    srch_saturday_night_bool INTEGER, random_bool INTEGER)
RETURNS INTEGER
AS $$
import numpy as np
import pandas as pd
import time
import onnxruntime as ort
import time

name = "pf2"

start = time.perf_counter()
onnx_path = '/home/public_datasets/Expedia/expedia_LR_pipeline.onnx'
ortconfig = ort.SessionOptions()
expedia_onnx_session = ort.InferenceSession(onnx_path, sess_options=ortconfig)
expedia_label = expedia_onnx_session.get_outputs()[0]
numerical_columns = ['prop_location_score1', 'prop_location_score2', 'prop_log_historical_price', 'price_usd',
                     'orig_destination_distance', 'prop_review_score', 'avg_bookings_usd', 'stdev_bookings_usd']
categorical_columns = ['position', 'prop_country_id', 'prop_starrating', 'prop_brand_bool', 'count_clicks',
                       'count_bookings', 'year', 'month', 'weekofyear', 'time', 'site_id',
                       'visitor_location_country_id', 'srch_destination_id', 'srch_length_of_stay',
                       'srch_booking_window', 'srch_adults_count', 'srch_children_count', 'srch_room_count',
                       'srch_saturday_night_bool', 'random_bool']
expedia_input_columns = numerical_columns + categorical_columns
expedia_type_map = {
    int: np.int64,
    bool: np.int64,
    float: np.float32,
    str: str
}
stop = time.perf_counter()
with open(f"/home/pg_init_{name}.log", 'a+') as f:
    f.write(f"{(stop-start)*1000}\n")

infer_batch = {
    elem: np.array([args[i]]).astype(expedia_type_map[type(args[i])]).reshape((-1, 1))
    for i, elem in enumerate(expedia_input_columns)
}
outputs = expedia_onnx_session.run([expedia_label.name], infer_batch)

return outputs[0][0]

$$ LANGUAGE plpython3u;


CREATE FUNCTION pf3(slatitude REAL, slongitude REAL, dlatitude REAL, dlongitude REAL, name1 INTEGER,
                    name2 VARCHAR, name4 VARCHAR, acountry VARCHAR, active_ VARCHAR, scity VARCHAR, scountry VARCHAR,
                    stimezone INTEGER, sdst VARCHAR, dcity VARCHAR, dcountry VARCHAR, dtimezone INTEGER, ddst VARCHAR)
RETURNS INTEGER
AS $$
import pickle
import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

name = "pf3"

scaler_path = '/home/public_datasets/Flights/flights_standard_scale_model.pkl'
enc_path = '/home/public_datasets/Flights/flights_one_hot_encoder.pkl'
model_path = '/home/public_datasets/Flights/flights_rf_model.pkl'
with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)
with open(enc_path, 'rb') as f:
    enc = pickle.load(f)
with open(model_path, 'rb') as f:
    model = pickle.load(f)

numerical = [args[0:4]]
categorical = [args[4:]]

X = np.hstack((scaler.transform(numerical), enc.transform(categorical).toarray()))
return model.predict(X)[0]

$$ LANGUAGE plpython3u;

CREATE FUNCTION pf4(slatitude REAL, slongitude REAL, dlatitude REAL, dlongitude REAL, name1 INTEGER,
                    name2 VARCHAR, name4 VARCHAR, acountry VARCHAR, active_ VARCHAR, scity VARCHAR, scountry VARCHAR,
                    stimezone INTEGER, sdst VARCHAR, dcity VARCHAR, dcountry VARCHAR, dtimezone INTEGER, ddst VARCHAR)
RETURNS INTEGER
AS $$
import numpy as np
import pandas as pd
import onnxruntime as ort

onnx_path = '/home/public_datasets/Flights/flights_rf_pipeline.onnx'
ortconfig = ort.SessionOptions()
flights_onnx_session = ort.InferenceSession(onnx_path, sess_options=ortconfig)
flights_label = flights_onnx_session.get_outputs()[0]
numerical_columns = ['slatitude', 'slongitude', 'dlatitude', 'dlongitude']
categorical_columns = ['name1', 'name2', 'name4', 'acountry', 'active', 'scity', 'scountry', 'stimezone', 'sdst',
                       'dcity', 'dcountry', 'dtimezone', 'ddst']
flights_input_columns = numerical_columns + categorical_columns
flights_type_map = {
    int: np.int64,
    bool: np.int64,
    float: np.float32,
    str: str
}

infer_batch = {
    elem: np.array(args[i]).astype(flights_type_map[type(args[i])]).reshape((-1, 1))
    for i, elem in enumerate(flights_input_columns)
}
outputs = flights_onnx_session.run([flights_label.name], infer_batch)
return outputs[0][0]

$$ LANGUAGE plpython3u;

CREATE FUNCTION pf5(hematocrit REAL, neutrophils REAL, sodium REAL, glucose REAL, bloodureanitro REAL,
                    creatinine REAL, bmi REAL, pulse INTEGER, respiration REAL, secondarydiagnosisnonicd9 INTEGER,
                    rcount VARCHAR, gender VARCHAR, dialysisrenalendstage INTEGER, asthma INTEGER, irondef INTEGER,
                    pneum INTEGER, substancedependence INTEGER, psychologicaldisordermajor INTEGER, depress INTEGER,
                    psychother INTEGER, fibrosisandother INTEGER, malnutrition INTEGER, hemo INTEGER)
RETURNS INTEGER
AS $$
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

name = "pf5"

scaler_path = '/home/public_datasets/Hospital/hospital_standard_scale_model.pkl'
enc_path = '/home/public_datasets/Hospital/hospital_one_hot_encoder.pkl'
model_path = '/home/public_datasets/Hospital/hospital_lr_model.pkl'
with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)
with open(enc_path, 'rb') as f:
    enc = pickle.load(f)
with open(model_path, 'rb') as f:
    model = pickle.load(f)

numerical = [args[0:10]]
categorical = [args[10:]]

X = np.hstack((scaler.transform(numerical), enc.transform(categorical).toarray()))
return model.predict(X)[0]
$$ LANGUAGE plpython3u;

CREATE FUNCTION pf6(hematocrit REAL, neutrophils REAL, sodium REAL, glucose REAL, bloodureanitro REAL,
                    creatinine REAL, bmi REAL, pulse INTEGER, respiration REAL, secondarydiagnosisnonicd9 INTEGER,
                    rcount VARCHAR, gender VARCHAR, dialysisrenalendstage INTEGER, asthma INTEGER, irondef INTEGER,
                    pneum INTEGER, substancedependence INTEGER, psychologicaldisordermajor INTEGER, depress INTEGER,
                    psychother INTEGER, fibrosisandother INTEGER, malnutrition INTEGER, hemo INTEGER)
RETURNS INTEGER
AS $$
import numpy as np
import pandas as pd
import onnxruntime as ort

name = "pf6"

onnx_path = '/home/public_datasets/Hospital/hospital_mlp_pipeline.onnx'
ortconfig = ort.SessionOptions()
hospital_onnx_session = ort.InferenceSession(onnx_path, sess_options=ortconfig)
hospital_label = hospital_onnx_session.get_outputs()[0]
numerical_columns = ['hematocrit', 'neutrophils', 'sodium', 'glucose', 'bloodureanitro', 'creatinine', 'bmi', 'pulse',
                     'respiration', 'secondarydiagnosisnonicd9']
categorical_columns = ['rcount', 'gender', 'dialysisrenalendstage', 'asthma', 'irondef', 'pneum', 'substancedependence',
                       'psychologicaldisordermajor', 'depress', 'psychother', 'fibrosisandother', 'malnutrition',
                       'hemo']
hospital_input_columns = numerical_columns + categorical_columns
hospital_type_map = {
    int: np.int64,
    bool: np.int64,
    float: np.float32,
    str: str
}

infer_batch = {
    elem: np.array(args[i]).astype(hospital_type_map[type(args[i])]).reshape((-1, 1))
    for i, elem in enumerate(hospital_input_columns )
}
outputs = hospital_onnx_session.run([hospital_label.name], infer_batch)
return outputs[0][0]

$$ LANGUAGE plpython3u;


CREATE FUNCTION pf7(V1 REAL, V2 REAL, V3 REAL, V4 REAL, V5 REAL, V6 REAL, V7 REAL, V8 REAL,
                    V9 REAL, V10 REAL, V11 REAL, V12 REAL, V13 REAL, V14 REAL, V15 REAL, V16 REAL,
                    V17 REAL, V18 REAL, V19 REAL, V20 REAL, V21 REAL, V22 REAL, V23 REAL, V24 REAL,
                    V25 REAL, V26 REAL, V27 REAL, V28 REAL, Amount REAL)
RETURNS INTEGER
AS $$
import pickle
import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

name = "pf7"

scaler_path = '/home/public_datasets/Credit_Card/creditcard_standard_scale_model.pkl'
model_path = '/home/public_datasets/Credit_Card/creditcard_lgb_model.txt'
with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)
model = lgb.Booster(model_file=model_path)

numerical = [args]

X = scaler.transform(numerical)
return round(model.predict(X)[0])

$$ LANGUAGE plpython3u;

CREATE FUNCTION pf8(V1 REAL, V2 REAL, V3 REAL, V4 REAL, V5 REAL, V6 REAL, V7 REAL, V8 REAL,
                    V9 REAL, V10 REAL, V11 REAL, V12 REAL, V13 REAL, V14 REAL, V15 REAL, V16 REAL,
                    V17 REAL, V18 REAL, V19 REAL, V20 REAL, V21 REAL, V22 REAL, V23 REAL, V24 REAL,
                    V25 REAL, V26 REAL, V27 REAL, V28 REAL, Amount REAL)
RETURNS INTEGER
AS $$
import pickle
import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier

name = "pf8"

scaler_path = '/home/public_datasets/Credit_Card/creditcard_standard_scale_model.pkl'
model_path = '/home/public_datasets/Credit_Card/creditcard_catboost_gb.cbm'
with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)
model = CatBoostClassifier()
model.load_model(model_path)

numerical = [args]

X = scaler.transform(numerical)
return round(model.predict(X)[0])

$$ LANGUAGE plpython3u;