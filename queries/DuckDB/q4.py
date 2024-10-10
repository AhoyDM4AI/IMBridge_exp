import pickle

import duckdb
import numpy as np
from duckdb.typing import BIGINT, DOUBLE, FLOAT, VARCHAR
import pandas as pd
import time
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from torch import nn
import torch

name = "q4"

con = duckdb.connect("imbridge2.db")

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

scaler_path = '/home/test_raven/Hospital/hospital_standard_scale_model.pkl'
enc_path = '/home/test_raven/Hospital/hospital_one_hot_encoder.pkl'
model_path = '/home/test_raven/Hospital/hospital_mlp_pytorch_2.model'
with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)
with open(enc_path, 'rb') as f:
    enc = pickle.load(f)

model = torch.load(model_path)
model.eval()


def udf(hematocrit, neutrophils, sodium, glucose, bloodureanitro, creatinine, bmi, pulse, respiration,
        secondarydiagnosisnonicd9,
        rcount, gender, dialysisrenalendstage, asthma, irondef, pneum, substancedependence, psychologicaldisordermajor,
        depress, psychother, fibrosisandothe, malnutrition, hemo):
    data = np.column_stack(
        [hematocrit, neutrophils, sodium, glucose, bloodureanitro, creatinine, bmi, pulse, respiration,
         secondarydiagnosisnonicd9,
         rcount, gender, dialysisrenalendstage, asthma, irondef, pneum, substancedependence,
         psychologicaldisordermajor, depress, psychother, fibrosisandothe, malnutrition, hemo])
    data = np.split(data, np.array([10]), axis=1)
    numerical = data[0]
    categorical = data[1]

    X = torch.tensor(np.hstack((scaler.transform(numerical), enc.transform(categorical).toarray())), dtype=torch.float32)
    with torch.no_grad():
        predictions = model(X)

    ret = predictions.numpy().reshape((1,-1))
    return ret[0]


con.create_function("udf", udf,
                    [DOUBLE, DOUBLE, DOUBLE, DOUBLE, DOUBLE, DOUBLE, DOUBLE, BIGINT,
                     DOUBLE, BIGINT, VARCHAR, VARCHAR,BIGINT, BIGINT,
                     BIGINT, BIGINT, BIGINT, BIGINT, BIGINT, BIGINT, BIGINT, BIGINT, BIGINT], BIGINT, type="arrow")

# con.sql("SET threads TO 1;")

con.sql('''
Explain analyze SELECT eid, udf(hematocrit, neutrophils, sodium, glucose, bloodureanitro, creatinine, bmi, pulse,
 respiration, secondarydiagnosisnonicd9, rcount, gender, cast(dialysisrenalendstage as INTEGER), cast(asthma as INTEGER),
  cast(irondef as INTEGER), cast(pneum as INTEGER), cast(substancedependence as INTEGER),
   cast(psychologicaldisordermajor as INTEGER), cast(depress as INTEGER), cast(psychother as INTEGER),
    cast(fibrosisandother as INTEGER), cast(malnutrition as INTEGER), cast(hemo as INTEGER)) AS lengthofstay
   FROM LengthOfStay_extension WHERE hematocrit > 10 AND neutrophils > 10 AND bloodureanitro < 20 AND pulse < 70;
''')

print(name)
