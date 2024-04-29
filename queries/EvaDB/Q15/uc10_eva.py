import joblib
import pandas as pd
import numpy as np
from evadb.catalog.catalog_type import NdArrayType

from evadb.functions.abstract.abstract_function import AbstractFunction
from evadb.functions.decorators.decorators import setup, forward
from evadb.functions.decorators.io_descriptors.data_types import PandasDataframe

from sklearn.linear_model import LogisticRegression

class uc10(AbstractFunction):
    @property
    def name(self) -> str:
        return "uc10"

    def setup(self):
        self.predict_col = self.name
        model_file_name = f"../../../tpcxai_datasets/sf10/model/{self.name}/{self.name}.python.model"
        self.model = joblib.load(model_file_name)

    @forward(input_signatures=[PandasDataframe(columns=["business_hour_norm", "amount_norm"],
                                               column_types=[NdArrayType.FLOAT64, NdArrayType.FLOAT64],
                                               column_shapes=[(None, ), (None, )])],
             output_signatures=[PandasDataframe(columns=["uc10"],
                                                column_types=[NdArrayType.INT64],
                                                column_shapes=[(None, )])])
    def forward(self, frames: pd.DataFrame) -> pd.DataFrame:
        #print(frames)
        #print(len(frames))
        result = self.model.predict(frames)
        predict_df = pd.DataFrame({self.predict_col: result})
        return predict_df
