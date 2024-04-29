CREATE PYTHON_UDF tpcx_ai_uc08_holistic(scan_count INTEGER, scan_count_abs INTEGER, Monday INTEGER, Tuesday INTEGER, Wednesday INTEGER, Thursday INTEGER, Friday INTEGER, Saturday INTEGER, Sunday INTEGER, dep0 INTEGER, dep1 INTEGER, dep2 INTEGER, dep3 INTEGER, dep4 INTEGER, dep5 INTEGER, dep6 INTEGER, dep7 INTEGER, dep8 INTEGER, dep9 INTEGER, dep10 INTEGER, dep11 INTEGER, dep12 INTEGER, dep13 INTEGER, dep14 INTEGER, dep15 INTEGER, dep16 INTEGER, dep17 INTEGER, dep18 INTEGER, dep19 INTEGER, dep20 INTEGER, dep21 INTEGER, dep22 INTEGER, dep23 INTEGER, dep24 INTEGER, dep25 INTEGER, dep26 INTEGER, dep27 INTEGER, dep28 INTEGER, dep29 INTEGER, dep30 INTEGER, dep31 INTEGER, dep32 INTEGER, dep33 INTEGER, dep34 INTEGER, dep35 INTEGER, dep36 INTEGER, dep37 INTEGER, dep38 INTEGER, dep39 INTEGER, dep40 INTEGER, dep41 INTEGER, dep42 INTEGER, dep43 INTEGER, dep44 INTEGER, dep45 INTEGER, dep46 INTEGER, dep47 INTEGER, dep48 INTEGER, dep49 INTEGER, dep50 INTEGER, dep51 INTEGER, dep52 INTEGER, dep53 INTEGER, dep54 INTEGER, dep55 INTEGER, dep56 INTEGER, dep57 INTEGER, dep58 INTEGER, dep59 INTEGER, dep60 INTEGER, dep61 INTEGER, dep62 INTEGER, dep63 INTEGER, dep64 INTEGER, dep65 INTEGER, dep66 INTEGER, dep67 INTEGER) RETURNS INTEGER 
{"
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
    pass

def pyfun(*args):
    # Inference context setup
    model_file = '{HOME_PATH}/workload/tpcxai_datasets/sf10/models/uc08/uc08.python.model'
    uc08_model = joblib.load(model_file)
    # Data preprocess
    data = pd.DataFrame({'scan_count': args[0], 'scan_count_abs': args[1], 'Monday': args[2], 'Tuesday': args[3], 'Wednesday': args[4], 'Thursday': args[5], 'Friday': args[6], 'Saturday': args[7], 'Sunday': args[8], 'dep0': args[9], 'dep1': args[10], 'dep2': args[11], 'dep3': args[12], 'dep4': args[13], 'dep5': args[14], 'dep6': args[15], 'dep7': args[16], 'dep8': args[17], 'dep9': args[18], 'dep10': args[19], 'dep11': args[20], 'dep12': args[21], 'dep13': args[22], 'dep14': args[23], 'dep15': args[24], 'dep16': args[25], 'dep17': args[26], 'dep18': args[27], 'dep19': args[28], 'dep20': args[29], 'dep21': args[30], 'dep22': args[31], 'dep23': args[32], 'dep24': args[33], 'dep25': args[34], 'dep26': args[35], 'dep27': args[36], 'dep28': args[37], 'dep29': args[38], 'dep30': args[39], 'dep31': args[40], 'dep32': args[41], 'dep33': args[42], 'dep34': args[43], 'dep35': args[44], 'dep36': args[45], 'dep37': args[46], 'dep38': args[47], 'dep39': args[48], 'dep40': args[49], 'dep41': args[50], 'dep42': args[51], 'dep43': args[52], 'dep44': args[53], 'dep45': args[54], 'dep46': args[55], 'dep47': args[56], 'dep48': args[57], 'dep49': args[58], 'dep50': args[59], 'dep51': args[60], 'dep52': args[61], 'dep53': args[62], 'dep54': args[63], 'dep55': args[64], 'dep56': args[65], 'dep57': args[66], 'dep58': args[67], 'dep59': args[68], 'dep60': args[69], 'dep61': args[70], 'dep62': args[71], 'dep63': args[72], 'dep64': args[73], 'dep65': args[74], 'dep66': args[75], 'dep67': args[76]})
    
    sparse_data = csr_matrix(data)
    # Invoke model inference
    predictions = uc08_model.predict(sparse_data)
    dec_fun = np.vectorize(decode_label)

    return dec_fun(predictions)
"};

CREATE PYTHON_UDF tpcx_ai_uc08_staged(scan_count INTEGER, scan_count_abs INTEGER, Monday INTEGER, Tuesday INTEGER, Wednesday INTEGER, Thursday INTEGER, Friday INTEGER, Saturday INTEGER, Sunday INTEGER, dep0 INTEGER, dep1 INTEGER, dep2 INTEGER, dep3 INTEGER, dep4 INTEGER, dep5 INTEGER, dep6 INTEGER, dep7 INTEGER, dep8 INTEGER, dep9 INTEGER, dep10 INTEGER, dep11 INTEGER, dep12 INTEGER, dep13 INTEGER, dep14 INTEGER, dep15 INTEGER, dep16 INTEGER, dep17 INTEGER, dep18 INTEGER, dep19 INTEGER, dep20 INTEGER, dep21 INTEGER, dep22 INTEGER, dep23 INTEGER, dep24 INTEGER, dep25 INTEGER, dep26 INTEGER, dep27 INTEGER, dep28 INTEGER, dep29 INTEGER, dep30 INTEGER, dep31 INTEGER, dep32 INTEGER, dep33 INTEGER, dep34 INTEGER, dep35 INTEGER, dep36 INTEGER, dep37 INTEGER, dep38 INTEGER, dep39 INTEGER, dep40 INTEGER, dep41 INTEGER, dep42 INTEGER, dep43 INTEGER, dep44 INTEGER, dep45 INTEGER, dep46 INTEGER, dep47 INTEGER, dep48 INTEGER, dep49 INTEGER, dep50 INTEGER, dep51 INTEGER, dep52 INTEGER, dep53 INTEGER, dep54 INTEGER, dep55 INTEGER, dep56 INTEGER, dep57 INTEGER, dep58 INTEGER, dep59 INTEGER, dep60 INTEGER, dep61 INTEGER, dep62 INTEGER, dep63 INTEGER, dep64 INTEGER, dep65 INTEGER, dep66 INTEGER, dep67 INTEGER) RETURNS INTEGER 
{"
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

# Execute at the PF Initialization Stage
def pyinitial():
    # Inference context setup
    global uc08_model
    model_file = '{HOME_PATH}/workload/tpcxai_datasets/sf10/models/uc08/uc08.python.model'
    uc08_model = joblib.load(model_file)

# Execute at the PF Computation Stage
def pyfun(*args):
    # Data preprocess
    data = pd.DataFrame({'scan_count': args[0], 'scan_count_abs': args[1], 'Monday': args[2], 'Tuesday': args[3], 'Wednesday': args[4], 'Thursday': args[5], 'Friday': args[6], 'Saturday': args[7], 'Sunday': args[8], 'dep0': args[9], 'dep1': args[10], 'dep2': args[11], 'dep3': args[12], 'dep4': args[13], 'dep5': args[14], 'dep6': args[15], 'dep7': args[16], 'dep8': args[17], 'dep9': args[18], 'dep10': args[19], 'dep11': args[20], 'dep12': args[21], 'dep13': args[22], 'dep14': args[23], 'dep15': args[24], 'dep16': args[25], 'dep17': args[26], 'dep18': args[27], 'dep19': args[28], 'dep20': args[29], 'dep21': args[30], 'dep22': args[31], 'dep23': args[32], 'dep24': args[33], 'dep25': args[34], 'dep26': args[35], 'dep27': args[36], 'dep28': args[37], 'dep29': args[38], 'dep30': args[39], 'dep31': args[40], 'dep32': args[41], 'dep33': args[42], 'dep34': args[43], 'dep35': args[44], 'dep36': args[45], 'dep37': args[46], 'dep38': args[47], 'dep39': args[48], 'dep40': args[49], 'dep41': args[50], 'dep42': args[51], 'dep43': args[52], 'dep44': args[53], 'dep45': args[54], 'dep46': args[55], 'dep47': args[56], 'dep48': args[57], 'dep49': args[58], 'dep50': args[59], 'dep51': args[60], 'dep52': args[61], 'dep53': args[62], 'dep54': args[63], 'dep55': args[64], 'dep56': args[65], 'dep57': args[66], 'dep58': args[67], 'dep59': args[68], 'dep60': args[69], 'dep61': args[70], 'dep62': args[71], 'dep63': args[72], 'dep64': args[73], 'dep65': args[74], 'dep66': args[75], 'dep67': args[76]})

    sparse_data = csr_matrix(data)
    # Invoke model inference
    predictions = uc08_model.predict(sparse_data)
    dec_fun = np.vectorize(decode_label)

    return dec_fun(predictions)
"};