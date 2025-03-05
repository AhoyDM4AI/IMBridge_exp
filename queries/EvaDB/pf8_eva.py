import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from evadb.catalog.catalog_type import NdArrayType

from evadb.functions.abstract.abstract_function import AbstractFunction
from evadb.functions.decorators.decorators import setup, forward
from evadb.functions.decorators.io_descriptors.data_types import PandasDataframe



class pf8(AbstractFunction):
    @property
    def name(self) -> str:
        return "pf8"

    def setup(self):
        self.predict_col = self.name
        scaler_path = '/home/test_raven/Credit_Card/creditcard_standard_scale_model.pkl'
        model_path = '/home/test_raven/Credit_Card/creditcard_xgb_model.json'
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        self.model = xgb.Booster(model_file=model_path)

    @forward(input_signatures=[PandasDataframe(columns=["V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10", "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "Amount"],
                                                column_types=[NdArrayType.FLOAT64, NdArrayType.FLOAT64, NdArrayType.FLOAT64, NdArrayType.FLOAT64, NdArrayType.FLOAT64, NdArrayType.FLOAT64, NdArrayType.FLOAT64, NdArrayType.FLOAT64, NdArrayType.FLOAT64, NdArrayType.FLOAT64, NdArrayType.FLOAT64, NdArrayType.FLOAT64, NdArrayType.FLOAT64, NdArrayType.FLOAT64, NdArrayType.FLOAT64, NdArrayType.FLOAT64, NdArrayType.FLOAT64, NdArrayType.FLOAT64, NdArrayType.FLOAT64, NdArrayType.FLOAT64, NdArrayType.FLOAT64, NdArrayType.FLOAT64, NdArrayType.FLOAT64, NdArrayType.FLOAT64, NdArrayType.FLOAT64, NdArrayType.FLOAT64, NdArrayType.FLOAT64, NdArrayType.FLOAT64, NdArrayType.FLOAT64],
                                                column_shapes=[(None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, )])],
                                                output_signatures=[PandasDataframe(columns=["pf8"],
                                                column_types=[NdArrayType.INT64],
                                                column_shapes=[(None, )])])
    def forward(self, frames: pd.DataFrame) -> pd.DataFrame:
        X = self.scaler.transform(frames.to_numpy())
        predictions = self.model.predict(xgb.DMatrix(X))
        predict_df = pd.DataFrame({self.predict_col: predictions})
        return predict_df
