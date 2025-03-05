import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, LabelBinarizer
import lightgbm as lgb

from evadb.catalog.catalog_type import NdArrayType

from evadb.functions.abstract.abstract_function import AbstractFunction
from evadb.functions.decorators.decorators import setup, forward
from evadb.functions.decorators.io_descriptors.data_types import PandasDataframe

class tpch10(AbstractFunction):
    @property
    def name(self) -> str:
        return "tpch10"

    def setup(self):
        self.predict_col = self.name
        scaler_path = '/home/test_tpch/Q10_standard_scale_model.pkl'
        enc_path = '/home/test_tpch/Q10_one_hot_encoder.pkl'
        lb_path = '/home/test_tpch/Q10_label_binarizer.pkl'
        model_path = '/home/test_tpch/Q10_lgb_gbdt_model.txt'
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        with open(enc_path, 'rb') as f:
            self.enc = pickle.load(f)
        with open(lb_path, 'rb') as f:
            self.lb = pickle.load(f)

        self.model = lgb.Booster(model_file=model_path)

    @forward(input_signatures=[PandasDataframe(columns=["c_acctbal", "o_totalprice", "l_quantity", "l_extendedprice", "l_discount", "l_tax",
                                                        "o_orderstatus", "o_orderpriority", "l_linestatus", "l_shipinstruct", "l_shipmode", "n_nationkey", "n_regionkey"],
                                               column_types=[NdArrayType.FLOAT64, NdArrayType.FLOAT64, NdArrayType.FLOAT64, NdArrayType.FLOAT64, 
                                                             NdArrayType.FLOAT64, NdArrayType.FLOAT64, NdArrayType.STR,
                                                               NdArrayType.STR, NdArrayType.STR, NdArrayType.STR, NdArrayType.STR, NdArrayType.FLOAT64, NdArrayType.FLOAT64],
                                               column_shapes=[(None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, )])],
             output_signatures=[PandasDataframe(columns=["tpch10"],
                                                column_types=[NdArrayType.STR],
                                                column_shapes=[(None, )])])
    def forward(self, frames: pd.DataFrame) -> pd.DataFrame:
        data = np.split(frames.to_numpy(), np.array([6]), axis = 1)
        numerical = data[0]
        categorical = data[1]
        X = np.hstack((self.scaler.transform(numerical), self.enc.transform(categorical).toarray()))
        res = self.lb.inverse_transform(self.model.predict(X))
        predict_df = pd.DataFrame({self.predict_col: res})
        return predict_df
