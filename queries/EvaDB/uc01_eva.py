import joblib
import pandas as pd
from evadb.catalog.catalog_type import NdArrayType

from evadb.functions.abstract.abstract_function import AbstractFunction
from evadb.functions.decorators.decorators import setup, forward
from evadb.functions.decorators.io_descriptors.data_types import PandasDataframe
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler


class GenericUCModel(AbstractFunction):
    @property
    def name(self) -> str:
        return "uc01"

    def setup(self):
        self.predict_col = self.name
        model_file_name = f"/home/model/{self.name}/{self.name}.python.model"
        self.model = joblib.load(model_file_name)

    @forward(input_signatures=[PandasDataframe(columns=["return_ratio", "frequency"],
                                               column_types=[NdArrayType.FLOAT32, NdArrayType.FLOAT32],
                                               column_shapes=[(None, ), (None, )])],
             output_signatures=[PandasDataframe(columns=["uc01"],
                                                column_types=[NdArrayType.INT64],
                                                column_shapes=[(None, )])])
    def forward(self, frames: pd.DataFrame) -> pd.DataFrame:
        predictions = self.model.predict(frames)
        print(len(frames))
        predict_df = pd.DataFrame(predictions)
        predict_df.rename(columns={0: self.predict_col}, inplace=True)
        return predict_df
