CREATE PYTHON_UDF expedia_sklearn_dt_holistic(prop_location_score1 REAL, prop_location_score2 REAL, prop_log_historical_price REAL, price_usd REAL, orig_destination_distance REAL, prop_review_score REAL, avg_bookings_usd REAL, stdev_bookings_usd REAL, position_ STRING, prop_country_id STRING, prop_starrating INTEGER, prop_brand_bool INTEGER, count_clicks INTEGER, count_bookings INTEGER, year_ STRING, month_ STRING, weekofyear_ STRING, time_ STRING, site_id STRING, visitor_location_country_id STRING, srch_destination_id STRING, srch_length_of_stay INTEGER, srch_booking_window INTEGER, srch_adults_count INTEGER, srch_children_count INTEGER, srch_room_count INTEGER, srch_saturday_night_bool INTEGER, random_bool INTEGER) RETURNS INTEGER 
{"
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
def pyinitial():
    pass
def pyfun(*args):
    # Inference context setup
    scaler_path = '{HOME_PATH}/workload/raven_datasets/Expedia/expedia_standard_scale_model.pkl'
    enc_path = '{HOME_PATH}/workload/raven_datasets/Expedia/expedia_one_hot_encoder.pkl'
    model_path = '{HOME_PATH}/workload/raven_datasets/Expedia/expedia_dt_model.pkl'
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    with open(enc_path, 'rb') as f:
        enc = pickle.load(f)
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    # Data preprocess
    data = np.column_stack(args)
    data = np.split(data, np.array([8]), axis = 1)
    numerical = data[0]
    categorical = data[1]
    X = np.hstack((scaler.transform(numerical), enc.transform(categorical).toarray()))
    # Invoke model inference
    return model.predict(X)
"};


CREATE PYTHON_UDF expedia_sklearn_dt_staged(prop_location_score1 REAL, prop_location_score2 REAL, prop_log_historical_price REAL, price_usd REAL, orig_destination_distance REAL, prop_review_score REAL, avg_bookings_usd REAL, stdev_bookings_usd REAL, position_ STRING, prop_country_id STRING, prop_starrating INTEGER, prop_brand_bool INTEGER, count_clicks INTEGER, count_bookings INTEGER, year_ STRING, month_ STRING, weekofyear_ STRING, time_ STRING, site_id STRING, visitor_location_country_id STRING, srch_destination_id STRING, srch_length_of_stay INTEGER, srch_booking_window INTEGER, srch_adults_count INTEGER, srch_children_count INTEGER, srch_room_count INTEGER, srch_saturday_night_bool INTEGER, random_bool INTEGER) RETURNS INTEGER 
{"
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
# Execute at the PF Initialization Stage
def pyinitial():
    # Inference context setup
    global expedia_scaler, expedia_enc, expedia_model
    scaler_path = '{HOME_PATH}/workload/raven_datasets/Expedia/expedia_standard_scale_model.pkl'
    enc_path = '{HOME_PATH}/workload/raven_datasets/Expedia/expedia_one_hot_encoder.pkl'
    model_path = '{HOME_PATH}/workload/raven_datasets/Expedia/expedia_dt_model.pkl'
    with open(scaler_path, 'rb') as f:
        expedia_scaler = pickle.load(f)
    with open(enc_path, 'rb') as f:
        expedia_enc = pickle.load(f)
    with open(model_path, 'rb') as f:
        expedia_model = pickle.load(f)
# Execute at the PF Computation Stage
def pyfun(*args):
    # Data preprocess
    data = np.column_stack(args)
    data = np.split(data, np.array([8]), axis = 1)
    numerical = data[0]
    categorical = data[1]
    X = np.hstack((expedia_scaler.transform(numerical), expedia_enc.transform(categorical).toarray()))
    # Invoke model inference
    return expedia_model.predict(X)
"};