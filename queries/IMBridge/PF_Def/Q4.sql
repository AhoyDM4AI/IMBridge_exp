CREATE PYTHON_UDF flights_onnx_rf_holistic(slatitude REAL, slongitude REAL, dlatitude REAL, dlongitude REAL, name1 INTEGER, name2 STRING, name4 STRING, acountry STRING, active_ STRING, scity STRING, scountry STRING, stimezone INTEGER, sdst STRING, dcity STRING, dcountry STRING, dtimezone INTEGER, ddst STRING) RETURNS INTEGER 
{"
import numpy as np
import pandas as pd
import onnxruntime as ort
# import time
def pyinitial():
    pass
def pyfun(*args):
    # Inference context setup
    onnx_path = '{HOME_PATH}/workload/raven_datasets/Flights/flights_rf_pipeline.onnx'
    ortconfig = ort.SessionOptions()
    onnx_session = ort.InferenceSession(onnx_path, sess_options=ortconfig)
    label = onnx_session.get_outputs()[0]
    numerical_columns = ['slatitude', 'slongitude', 'dlatitude', 'dlongitude']
    categorical_columns = ['name1', 'name2', 'name4', 'acountry', 'active',
	'scity', 'scountry', 'stimezone', 'sdst',
	'dcity', 'dcountry', 'dtimezone', 'ddst']
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

CREATE PYTHON_UDF flights_onnx_rf_staged(slatitude REAL, slongitude REAL, dlatitude REAL, dlongitude REAL, name1 INTEGER, name2 STRING, name4 STRING, acountry STRING, active_ STRING, scity STRING, scountry STRING, stimezone INTEGER, sdst STRING, dcity STRING, dcountry STRING, dtimezone INTEGER, ddst STRING) RETURNS INTEGER 
{"
import numpy as np
import pandas as pd
import onnxruntime as ort
# Execute at the PF Initialization Stage
def pyinitial():
    # Inference context setup
    global flights_onnx_session, flights_type_map, flights_input_columns, flights_label
    onnx_path = '{HOME_PATH}/workload/raven_datasets/Flights/flights_rf_pipeline.onnx'
    ortconfig = ort.SessionOptions()
    flights_onnx_session = ort.InferenceSession(onnx_path, sess_options=ortconfig)
    flights_label = flights_onnx_session.get_outputs()[0]
    numerical_columns = ['slatitude', 'slongitude', 'dlatitude', 'dlongitude']
    categorical_columns = ['name1', 'name2', 'name4', 'acountry', 'active', 'scity', 'scountry', 'stimezone', 'sdst', 'dcity', 'dcountry', 'dtimezone', 'ddst']
    flights_input_columns = numerical_columns + categorical_columns
    flights_type_map = {
    'int32': np.int64,
    'int64': np.int64,
    'float64': np.float32,
    'object': str,
    }
# Execute at the PF Computation Stage
def pyfun(*args):
    # Data preprocess
    infer_batch = {
        elem: args[i].astype(flights_type_map[args[i].dtype.name]).reshape((-1, 1))
        for i, elem in enumerate(flights_input_columns)
    }
    # Invoke model inference
    outputs = flights_onnx_session.run([flights_label.name], infer_batch)
    return outputs[0]
"};