import joblib
import pandas as pd
import numpy as np
from evadb.catalog.catalog_type import NdArrayType

from evadb.functions.abstract.abstract_function import AbstractFunction
from evadb.functions.decorators.decorators import setup, forward
from evadb.functions.decorators.io_descriptors.data_types import PandasDataframe

from surprise import SVD
from surprise import Dataset
from surprise.reader import Reader

class uc07(AbstractFunction):
    @property
    def name(self) -> str:
        return "uc07"

    def setup(self):
        self.predict_col = self.name
        model_file_name = f"/home/model/{self.name}/{self.name}.python.model"
        self.model = joblib.load(model_file_name)

    @forward(input_signatures=[PandasDataframe(columns=["userId", "productId"],
                                               column_types=[NdArrayType.INT64, NdArrayType.INT64],
                                               column_shapes=[(None, ), (None, )])],
             output_signatures=[PandasDataframe(columns=["uc07"],
                                                column_types=[NdArrayType.FLOAT64],
                                                column_shapes=[(None, )])])
    def forward(self, frames: pd.DataFrame) -> pd.DataFrame:
        #print(frames)
        print(len(frames))
        user_id = frames['userid']
        item_id = frames['productid']
        ratings = []
        for i in range(len(user_id)):
            rating = self.model.predict(user_id[i], item_id[i]).est
            ratings.append(rating)
        predict_df = pd.DataFrame({self.predict_col: ratings})
        return predict_df
