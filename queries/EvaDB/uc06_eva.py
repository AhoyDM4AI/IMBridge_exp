import joblib
import pandas as pd
import numpy as np
from evadb.catalog.catalog_type import NdArrayType

from evadb.functions.abstract.abstract_function import AbstractFunction
from evadb.functions.decorators.decorators import setup, forward
from evadb.functions.decorators.io_descriptors.data_types import PandasDataframe

from imblearn.over_sampling import ADASYN
from imblearn.pipeline import Pipeline as Imb_Pipeline
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

class uc06(AbstractFunction):
    @property
    def name(self) -> str:
        return "uc06"

    def setup(self):
        self.predict_col = self.name
        model_file_name = f"/home/model/{self.name}/{self.name}.python.model"
        self.model = joblib.load(model_file_name)

    @forward(input_signatures=[PandasDataframe(columns=["smart_5_raw", "smart_10_raw", "smart_184_raw", "smart_187_raw", "smart_188_raw", "smart_197_raw", "smart_198_raw"],
                                               column_types=[NdArrayType.FLOAT64, NdArrayType.FLOAT64, NdArrayType.FLOAT64, NdArrayType.FLOAT64, NdArrayType.FLOAT64, NdArrayType.FLOAT64, NdArrayType.FLOAT64],
                                               column_shapes=[(None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, )])],
             output_signatures=[PandasDataframe(columns=["uc06"],
                                                column_types=[NdArrayType.INT64],
                                                column_shapes=[(None, )])])
    def forward(self, frames: pd.DataFrame) -> pd.DataFrame:
        predictions = self.model.predict(frames)
        print(len(frames))
        predict_df = pd.DataFrame({self.predict_col: predictions})
        #predict_df.rename(columns={0:self.predict_col}, inplace=True)
        return predict_df
