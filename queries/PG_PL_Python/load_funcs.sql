CREATE FUNCTION udf1(prop_location_score1 REAL, prop_location_score2 REAL,
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

name = "udf1"

onnx_path = '/home/test_raven/Expedia/expedia_dt_pipeline.onnx'
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

infer_batch = {
    elem: np.array([args[i]]).astype(expedia_type_map[type(args[i])]).reshape((-1, 1))
    for i, elem in enumerate(expedia_input_columns)
}
outputs = expedia_onnx_session.run([expedia_label.name], infer_batch)

return outputs[0][0]

$$ LANGUAGE plpython3u PARALLEL SAFE;


CREATE FUNCTION udf2(prop_location_score1 REAL, prop_location_score2 REAL,
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

name = "udf2"

scaler_path = '/home/test_raven/Expedia/expedia_standard_scale_model.pkl'
enc_path = '/home/test_raven/Expedia/expedia_one_hot_encoder.pkl'
model_path = '/home/test_raven/Expedia/expedia_dt_model.pkl'
with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)
with open(enc_path, 'rb') as f:
    enc = pickle.load(f)
with open(model_path, 'rb') as f:
    model = pickle.load(f)

numerical = [args[0:8]]
categorical = [args[8:]]

X = np.hstack((scaler.transform(numerical), enc.transform(categorical).toarray()))
return model.predict(X)[0]
$$ LANGUAGE plpython3u PARALLEL SAFE;

CREATE FUNCTION udf3(slatitude REAL, slongitude REAL, dlatitude REAL, dlongitude REAL, name1 INTEGER,
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

name = "udf3"

scaler_path = '/home/test_raven/Flights/flights_standard_scale_model.pkl'
enc_path = '/home/test_raven/Flights/flights_one_hot_encoder.pkl'
model_path = '/home/test_raven/Flights/flights_rf_model.pkl'
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

$$ LANGUAGE plpython3u PARALLEL SAFE;


CREATE FUNCTION udf4(hematocrit REAL, neutrophils REAL, sodium REAL, glucose REAL, bloodureanitro REAL,
                    creatinine REAL, bmi REAL, pulse INTEGER, respiration REAL, secondarydiagnosisnonicd9 INTEGER,
                    rcount VARCHAR, gender VARCHAR, dialysisrenalendstage INTEGER, asthma INTEGER, irondef INTEGER,
                    pneum INTEGER, substancedependence INTEGER, psychologicaldisordermajor INTEGER, depress INTEGER,
                    psychother INTEGER, fibrosisandother INTEGER, malnutrition INTEGER, hemo INTEGER)
RETURNS INTEGER
AS $$
import numpy as np
import pandas as pd
from torch import nn
import pickle
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

import torch
main = __import__('__main__')

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

name = "udf4"
main.MyModel = MyModel

scaler_path = '/home/test_raven/Hospital/hospital_standard_scale_model.pkl'
enc_path = '/home/test_raven/Hospital/hospital_one_hot_encoder.pkl'
model_path = '/home/test_raven/Hospital/hospital_mlp_pytorch_2.model'

with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)
with open(enc_path, 'rb') as f:
    enc = pickle.load(f)

model = torch.load(model_path)
model.eval()

numerical = [args[0:10]]
categorical = [args[10:]]

X = torch.tensor(np.hstack((scaler.transform(numerical), enc.transform(categorical).toarray())), dtype=torch.float32)
with torch.no_grad():
    predictions = model(X)

ret = predictions.numpy().reshape((1,-1))
return round(ret[0][0])
$$ LANGUAGE plpython3u PARALLEL SAFE;


CREATE FUNCTION udf5(V1 REAL, V2 REAL, V3 REAL, V4 REAL, V5 REAL, V6 REAL, V7 REAL, V8 REAL,
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

name = "udf5"

scaler_path = '/home/test_raven/Credit_Card/creditcard_standard_scale_model.pkl'
model_path = '/home/test_raven/Credit_Card/creditcard_lgb_model.txt'
with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)
model = lgb.Booster(model_file=model_path)

numerical = [args]

X = scaler.transform(numerical)
return round(model.predict(X)[0])

$$ LANGUAGE plpython3u PARALLEL SAFE;


CREATE FUNCTION udf6(V1 REAL, V2 REAL, V3 REAL, V4 REAL, V5 REAL, V6 REAL, V7 REAL, V8 REAL,
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
import xgboost as xgb

name = "udf5"

scaler_path = '/home/test_raven/Credit_Card/creditcard_standard_scale_model.pkl'
model_path = '/home/test_raven/Credit_Card/creditcard_xgb_model.json'
with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)
model = xgb.Booster(model_file=model_path)

numerical = [args]

X = scaler.transform(numerical)
return round(model.predict(xgb.DMatrix(X))[0])

$$ LANGUAGE plpython3u PARALLEL SAFE;

CREATE FUNCTION udf7(return_ratio REAL, frequency REAL)
RETURNS INTEGER
AS $$
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

name = "udf7"
mname = "uc01"
model_file_name = f"/home/model/{mname}/{mname}.python.model"

model = joblib.load(model_file_name)
feat = pd.DataFrame({
    'return_ratio': np.array([return_ratio]),
    'frequency': np.array([frequency])
})

return model.predict(feat)[0]
$$ LANGUAGE plpython3u PARALLEL SAFE;

CREATE FUNCTION udf8(store text, department text)
RETURNS text
AS $$
import warnings
import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import ValueWarning
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import joblib

name = "udf8"
mname = "uc03"
model_file_name = f"/home/model/{mname}/{mname}.python.model"

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
$$ LANGUAGE plpython3u PARALLEL SAFE;

CREATE FUNCTION udf9(txt text)
RETURNS INTEGER
AS $$
import joblib
import numpy as np
import pandas as pd
from sklearn import feature_extraction, naive_bayes
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer

name = "udf9"
mname = "uc04"
model_file_name = f"/home/model/{mname}/{mname}.python.model"

model = joblib.load(model_file_name)
data = pd.DataFrame({
    "text": np.array([txt])
})
return model.predict(data["text"])[0]

$$ LANGUAGE plpython3u PARALLEL SAFE;



CREATE FUNCTION udf10(smart_5_raw REAL,smart_10_raw REAL,smart_184_raw REAL,
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

name = "udf10"
mname = "uc06"
model_file_name = f"/home/model/{mname}/{mname}.python.model"

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

$$ LANGUAGE plpython3u PARALLEL SAFE;

CREATE FUNCTION udf11(user_id INTEGER, item_id INTEGER)
RETURNS REAL
AS $$
import numpy as np
import pandas as pd
import joblib
from surprise import SVD
from surprise import Dataset
from surprise.reader import Reader

name = "udf11"
mname = "uc07"
model_file_name = f"/home/model/{mname}/{mname}.python.model"

model = joblib.load(model_file_name)
rating = model.predict(user_id, item_id).est
return rating
$$ LANGUAGE plpython3u PARALLEL SAFE;


CREATE FUNCTION udf12(business_hour_norm REAL, amount_norm REAL)
RETURNS REAL
AS $$
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

name = "udf12"
mname = "uc10"
model_file_name = f"/home/model/{mname}/{mname}.python.model"

model = joblib.load(model_file_name)

data = pd.DataFrame({
    'business_hour_norm': np.array([business_hour_norm]),
    'amount_norm': np.array([amount_norm])
})
return model.predict(data)[0]

$$ LANGUAGE plpython3u PARALLEL SAFE;

