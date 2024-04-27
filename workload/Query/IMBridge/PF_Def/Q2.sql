CREATE PYTHON_UDF expedia_onnx_dt_holistic(prop_location_score1 REAL, prop_location_score2 REAL, prop_log_historical_price REAL, price_usd REAL, orig_destination_distance REAL, prop_review_score REAL, avg_bookings_usd REAL, stdev_bookings_usd REAL, position_ STRING, prop_country_id STRING, prop_starrating INTEGER, prop_brand_bool INTEGER, count_clicks INTEGER, count_bookings INTEGER, year_ STRING, month_ STRING, weekofyear_ STRING, time_ STRING, site_id STRING, visitor_location_country_id STRING, srch_destination_id STRING, srch_length_of_stay INTEGER, srch_booking_window INTEGER, srch_adults_count INTEGER, srch_children_count INTEGER, srch_room_count INTEGER, srch_saturday_night_bool INTEGER, random_bool INTEGER) RETURNS INTEGER 
{"
import numpy as np
import pandas as pd
import onnxruntime as ort
def pyinitial():
    pass
def pyfun(*args):
    # Inference context setup
    onnx_path = '{HOME_PATH}/workload/raven_datasets/Expedia/expedia_dt_pipeline.onnx'
    ortconfig = ort.SessionOptions()
    onnx_session = ort.InferenceSession(onnx_path, sess_options=ortconfig)
    label = onnx_session.get_outputs()[0]
    numerical_columns = ['prop_location_score1', 'prop_location_score2', 'prop_log_historical_price', 'price_usd', 'orig_destination_distance', 'prop_review_score', 'avg_bookings_usd', 'stdev_bookings_usd']
    categorical_columns = ['position', 'prop_country_id', 'prop_starrating', 'prop_brand_bool', 'count_clicks','count_bookings', 'year', 'month', 'weekofyear', 'time', 'site_id', 'visitor_location_country_id', 'srch_destination_id', 'srch_length_of_stay', 'srch_booking_window', 'srch_adults_count', 'srch_children_count', 'srch_room_count', 'srch_saturday_night_bool', 'random_bool']
    input_columns = numerical_columns + categorical_columns
    type_map = {
    'int32': np.int64,
    'int64': np.int64,
    'float64': np.float32,
    'object': str,
    }
    # Data preprocess
    infer_batch = {
        elem: args[i].astype(type_map[args[i].dtype.name]).reshape((-1, 1))
        for i, elem in enumerate(input_columns)
    }
    # Invoke model inference
    outputs = onnx_session.run([label.name], infer_batch)
    return outputs[0]
"};

CREATE PYTHON_UDF expedia_onnx_dt_staged(prop_location_score1 REAL, prop_location_score2 REAL, prop_log_historical_price REAL, price_usd REAL, orig_destination_distance REAL, prop_review_score REAL, avg_bookings_usd REAL, stdev_bookings_usd REAL, position_ STRING, prop_country_id STRING, prop_starrating INTEGER, prop_brand_bool INTEGER, count_clicks INTEGER, count_bookings INTEGER, year_ STRING, month_ STRING, weekofyear_ STRING, time_ STRING, site_id STRING, visitor_location_country_id STRING, srch_destination_id STRING, srch_length_of_stay INTEGER, srch_booking_window INTEGER, srch_adults_count INTEGER, srch_children_count INTEGER, srch_room_count INTEGER, srch_saturday_night_bool INTEGER, random_bool INTEGER) RETURNS INTEGER 
{"
import numpy as np
import pandas as pd
import onnxruntime as ort
# Execute at the PF Initialization Stage
def pyinitial():
    # Inference context setup
    global expedia_onnx_session, expedia_type_map, expedia_input_columns, expedia_label
    onnx_path = '{HOME_PATH}/workload/raven_datasets/Expedia/expedia_dt_pipeline.onnx'
    ortconfig = ort.SessionOptions()
    expedia_onnx_session = ort.InferenceSession(onnx_path, sess_options=ortconfig)
    expedia_label = expedia_onnx_session.get_outputs()[0]
    numerical_columns = ['prop_location_score1', 'prop_location_score2', 'prop_log_historical_price', 'price_usd', 'orig_destination_distance', 'prop_review_score', 'avg_bookings_usd', 'stdev_bookings_usd']
    categorical_columns = ['position', 'prop_country_id', 'prop_starrating', 'prop_brand_bool', 'count_clicks','count_bookings', 'year', 'month', 'weekofyear', 'time', 'site_id', 'visitor_location_country_id', 'srch_destination_id', 'srch_length_of_stay', 'srch_booking_window', 'srch_adults_count', 'srch_children_count', 'srch_room_count', 'srch_saturday_night_bool', 'random_bool']
    expedia_input_columns = numerical_columns + categorical_columns
    expedia_type_map = {
    'int32': np.int64,
    'int64': np.int64,
    'float64': np.float32,
    'object': str,
    }
# Execute at the PF Computation Stage
def pyfun(*args):
    # Data preprocess
    infer_batch = {
        elem: args[i].astype(expedia_type_map[args[i].dtype.name]).reshape((-1, 1))
        for i, elem in enumerate(expedia_input_columns)
    }
    # Invoke model inference
    outputs = expedia_onnx_session.run([expedia_label.name], infer_batch)
    return outputs[0]
"};