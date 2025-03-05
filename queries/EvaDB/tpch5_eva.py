import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, LabelBinarizer
import torch
from torch import nn
import pandas as pd

from evadb.catalog.catalog_type import NdArrayType

from evadb.functions.abstract.abstract_function import AbstractFunction
from evadb.functions.decorators.decorators import setup, forward
from evadb.functions.decorators.io_descriptors.data_types import PandasDataframe


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(None, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 5)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.softmax(self.fc3(x))
        return x

main = __import__("__main__")
main.MyModel = MyModel

class tpch5(AbstractFunction):
    @property
    def name(self) -> str:
        return "tpch5"

    def setup(self):
        self.predict_col = self.name
        scaler_path = '/home/test_tpch/Q5_standard_scale_model.pkl'
        enc_path = '/home/test_tpch/Q5_one_hot_encoder.pkl'
        lb_path = '/home/test_tpch/Q5_label_binarizer.pkl'
        model_path = '/home/test_tpch/Q5_pytorch_mlp.model'
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        with open(enc_path, 'rb') as f:
            self.enc = pickle.load(f)
        with open(lb_path, 'rb') as f:
            self.lb = pickle.load(f)
        self.mlp = torch.load(model_path)
        self.mlp.eval()

    @forward(input_signatures=[PandasDataframe(columns=["c_acctbal", "o_totalprice", "l_quantity", "l_extendedprice", "l_discount", "l_tax", "s_acctbal", "o_orderstatus", "l_returnflag", "l_linestatus", "l_shipinstruct", "l_shipmode", "n_nationkey", "n_regionkey"],
                                               column_types=[NdArrayType.FLOAT64, NdArrayType.FLOAT64, NdArrayType.FLOAT64, NdArrayType.FLOAT64, 
                                                             NdArrayType.FLOAT64, NdArrayType.FLOAT64, NdArrayType.FLOAT64, NdArrayType.STR,
                                                               NdArrayType.STR, NdArrayType.STR, NdArrayType.STR, NdArrayType.STR, NdArrayType.FLOAT64, NdArrayType.FLOAT64],
                                               column_shapes=[(None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, )])],
             output_signatures=[PandasDataframe(columns=["tpch5"],
                                                column_types=[NdArrayType.STR],
                                                column_shapes=[(None, )])])
    def forward(self, frames: pd.DataFrame) -> pd.DataFrame:
        data = np.split(frames.to_numpy(), np.array([7]), axis = 1)
        numerical = data[0]
        categorical = data[1]
        X = torch.tensor(np.hstack((self.scaler.transform(numerical), self.enc.transform(categorical).toarray())), dtype=torch.float32)
        with torch.no_grad():
            predictions = self.mlp(X)
        res = self.lb.inverse_transform(predictions.numpy())
        predict_df = pd.DataFrame({self.predict_col: res.reshape((1,-1))[0]})
        return predict_df
