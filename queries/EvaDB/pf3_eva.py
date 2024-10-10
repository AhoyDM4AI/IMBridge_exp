import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from evadb.catalog.catalog_type import NdArrayType

from evadb.functions.abstract.abstract_function import AbstractFunction
from evadb.functions.decorators.decorators import setup, forward
from evadb.functions.decorators.io_descriptors.data_types import PandasDataframe



class pf3(AbstractFunction):
    @property
    def name(self) -> str:
        return "pf3"

    def setup(self):
        self.predict_col = self.name
        scaler_path = '/home/test_raven/Flights/flights_standard_scale_model.pkl'
        enc_path = '/home/test_raven/Flights/flights_one_hot_encoder.pkl'
        model_path = '/home/test_raven/Flights/flights_rf_model.pkl'
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        with open(enc_path, 'rb') as f:
            self.enc = pickle.load(f)
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

    @forward(input_signatures=[PandasDataframe(columns=["slatitude", "slongitude", "dlatitude", "dlongitude", "name1", "name2", "name4", "acountry", "active", "scity", "scountry", "stimezone", "sdst", "dcity", "dcountry", "dtimezone", "ddst"],
                                               column_types=[NdArrayType.FLOAT64, NdArrayType.FLOAT64, NdArrayType.FLOAT64, NdArrayType.FLOAT64, NdArrayType.INT64, NdArrayType.STR, NdArrayType.STR, NdArrayType.STR, NdArrayType.STR, NdArrayType.STR, NdArrayType.STR, NdArrayType.INT64, NdArrayType.STR, NdArrayType.STR, NdArrayType.STR, NdArrayType.INT64, NdArrayType.STR],
                                               column_shapes=[(None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, )])],
             output_signatures=[PandasDataframe(columns=["pf3"],
                                                column_types=[NdArrayType.INT64],
                                                column_shapes=[(None, )])])
    def forward(self, frames: pd.DataFrame) -> pd.DataFrame:
        data = np.split(frames.to_numpy(), np.array([4]), axis = 1)
        numerical = data[0]
        categorical = data[1]
        X = np.hstack((self.scaler.transform(numerical), self.enc.transform(categorical).toarray()))
        predictions = self.model.predict(X)
        print(len(frames))
        predict_df = pd.DataFrame({self.predict_col: predictions})
        return predict_df
