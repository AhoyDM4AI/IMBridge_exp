import joblib
import pandas as pd
import numpy as np

from evadb.catalog.catalog_type import NdArrayType

from evadb.functions.abstract.abstract_function import AbstractFunction
from evadb.functions.decorators.decorators import setup, forward
from evadb.functions.decorators.io_descriptors.data_types import PandasDataframe

from sklearn import feature_extraction, naive_bayes
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer

class uc04(AbstractFunction):
    @property
    def name(self) -> str:
        return "uc04"

    def setup(self):
        self.predict_col = self.name
        model_file_name = f"../../../tpcxai_datasets/sf10/model/{self.name}/{self.name}.python.model"
        self.model = joblib.load(model_file_name)

    @forward(input_signatures=[PandasDataframe(columns=["text"],
                                               column_types=[NdArrayType.STR],
                                               column_shapes=[(None, )])],
             output_signatures=[PandasDataframe(columns=["uc04"],
                                                column_types=[NdArrayType.INT64],
                                                column_shapes=[(None, )])])
    def forward(self, frames: pd.DataFrame) -> pd.DataFrame:
        print(len(frames))
        predictions = self.model.predict(frames['text'].to_numpy().astype(str))
        predict_df = pd.DataFrame({self.predict_col: predictions})
        return predict_df
