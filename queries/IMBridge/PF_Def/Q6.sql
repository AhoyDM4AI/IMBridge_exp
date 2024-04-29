CREATE PYTHON_UDF hospital_onnx_lr_holistic(hematocrit REAL, neutrophils REAL, sodium REAL, glucose REAL, bloodureanitro REAL, creatinine REAL, bmi REAL, pulse INTEGER, respiration REAL, secondarydiagnosisnonicd9 INTEGER, rcount STRING, gender STRING, dialysisrenalendstage INTEGER, asthma INTEGER, irondef INTEGER, pneum INTEGER, substancedependence INTEGER, psychologicaldisordermajor INTEGER, depress INTEGER, psychother INTEGER, fibrosisandother INTEGER, malnutrition INTEGER, hemo INTEGER) RETURNS INTEGER 
{"
import numpy as np
import pandas as pd
import onnxruntime as ort
def pyinitial():
    pass
def pyfun(*args):
    # Inference context setup
    onnx_path = '{HOME_PATH}/workload/raven_datasets/Hospital/hospital_pipeline.onnx'
    ortconfig = ort.SessionOptions()
    onnx_session = ort.InferenceSession(onnx_path, sess_options=ortconfig)
    label = onnx_session.get_outputs()[0]
    numerical_columns = ['hematocrit', 'neutrophils', 'sodium', 'glucose', 'bloodureanitro', 'creatinine', 'bmi', 'pulse', 'respiration', 'secondarydiagnosisnonicd9']
    categorical_columns = ['rcount', 'gender', 'dialysisrenalendstage', 'asthma', 'irondef', 'pneum', 'substancedependence', 'psychologicaldisordermajor', 'depress', 'psychother', 'fibrosisandother', 'malnutrition', 'hemo']
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

CREATE PYTHON_UDF hospital_onnx_lr_staged(hematocrit REAL, neutrophils REAL, sodium REAL, glucose REAL, bloodureanitro REAL, creatinine REAL, bmi REAL, pulse INTEGER, respiration REAL, secondarydiagnosisnonicd9 INTEGER, rcount STRING, gender STRING, dialysisrenalendstage INTEGER, asthma INTEGER, irondef INTEGER, pneum INTEGER, substancedependence INTEGER, psychologicaldisordermajor INTEGER, depress INTEGER, psychother INTEGER, fibrosisandother INTEGER, malnutrition INTEGER, hemo INTEGER) RETURNS INTEGER  
{"
import numpy as np
import pandas as pd
import onnxruntime as ort
# Execute at the PF Initialization Stage
def pyinitial():
    # Inference context setup
    global hospital_onnx_session, hospital_type_map, hospital_input_columns, hospital_label
    onnx_path = '{HOME_PATH}/workload/raven_datasets/Hospital/hospital_pipeline.onnx'
    ortconfig = ort.SessionOptions()
    hospital_onnx_session = ort.InferenceSession(onnx_path, sess_options=ortconfig)
    hospital_label = hospital_onnx_session.get_outputs()[0]
    numerical_columns = ['hematocrit', 'neutrophils', 'sodium', 'glucose', 'bloodureanitro', 'creatinine', 'bmi', 'pulse', 'respiration', 'secondarydiagnosisnonicd9']
    categorical_columns = ['rcount', 'gender', 'dialysisrenalendstage', 'asthma', 'irondef', 'pneum', 'substancedependence', 'psychologicaldisordermajor', 'depress', 'psychother', 'fibrosisandother', 'malnutrition', 'hemo']
    hospital_input_columns = numerical_columns + categorical_columns
    hospital_type_map = {
    'int32': np.int64,
    'int64': np.int64,
    'float64': np.float32,
    'object': str,
    }
# Execute at the PF Computation Stage
def pyfun(*args):
    # Data preprocess
    infer_batch = {
        elem: args[i].astype(hospital_type_map[args[i].dtype.name]).reshape((-1, 1))
        for i, elem in enumerate(hospital_input_columns)
    }
    # Invoke model inference
    outputs = hospital_onnx_session.run([hospital_label.name], infer_batch)
    return outputs[0]
"};