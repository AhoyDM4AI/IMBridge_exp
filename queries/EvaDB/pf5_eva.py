import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from torch import nn
import torch
from evadb.catalog.catalog_type import NdArrayType

from evadb.functions.abstract.abstract_function import AbstractFunction
from evadb.functions.decorators.decorators import setup, forward
from evadb.functions.decorators.io_descriptors.data_types import PandasDataframe

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

main = __import__("__main__")
main.MyModel = MyModel

class pf5(AbstractFunction):
    @property
    def name(self) -> str:
        return "pf5"

    def setup(self):
        self.predict_col = self.name
        scaler_path = '/home/test_raven/Hospital/hospital_standard_scale_model.pkl'
        enc_path = '/home/test_raven/Hospital/hospital_one_hot_encoder.pkl'
        model_path = '/home/test_raven/Hospital/hospital_mlp_pytorch_2.model'
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        with open(enc_path, 'rb') as f:
            self.enc = pickle.load(f)

        self.model = torch.load(model_path)
        self.model.eval()

    @forward(input_signatures=[PandasDataframe(columns=["hematocrit", "neutrophils", "sodium", "glucose", "bloodureanitro", "creatinine", "bmi", "pulse", "respiration", "secondarydiagnosisnonicd9", "rcount", "gender", "dialysisrenalendstage", "asthma", "irondef", "pneum", "substancedependence", "psychologicaldisordermajor", "depress", "psychother", "fibrosisandother", "malnutrition", "hemo"],
                                                column_types=[NdArrayType.FLOAT64, NdArrayType.FLOAT64, NdArrayType.FLOAT64, NdArrayType.FLOAT64, NdArrayType.FLOAT64, NdArrayType.FLOAT64, NdArrayType.FLOAT64, NdArrayType.INT64, NdArrayType.FLOAT64, NdArrayType.INT64, NdArrayType.STR, NdArrayType.STR, NdArrayType.INT64, NdArrayType.INT64, NdArrayType.INT64, NdArrayType.INT64, NdArrayType.INT64, NdArrayType.INT64, NdArrayType.INT64, NdArrayType.INT64, NdArrayType.INT64, NdArrayType.INT64, NdArrayType.INT64],
                                                column_shapes=[(None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, )])],
                                                output_signatures=[PandasDataframe(columns=["pf5"],
                                                column_types=[NdArrayType.INT64],
                                                column_shapes=[(None, )])])
    def forward(self, frames: pd.DataFrame) -> pd.DataFrame:
        data = np.split(frames.to_numpy(), np.array([10]), axis = 1)
        numerical = data[0]
        categorical = data[1]
        X = torch.tensor(np.hstack((self.scaler.transform(numerical), self.enc.transform(categorical).toarray())), dtype=torch.float32)
        with torch.no_grad():
            predictions = self.model(X)
        predict_df = pd.DataFrame({self.predict_col: predictions.numpy().reshape((1,-1))[0]})
        return predict_df
