import joblib
import pandas as pd
import numpy as np
from evadb.catalog.catalog_type import NdArrayType

from evadb.functions.abstract.abstract_function import AbstractFunction
from evadb.functions.decorators.decorators import setup, forward
from evadb.functions.decorators.io_descriptors.data_types import PandasDataframe

from scipy.sparse import csr_matrix
from xgboost.sklearn import XGBClassifier

label_range = [3, 4, 5, 6, 7, 8, 9, 12, 15, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 999]
sorted_labels = sorted(label_range, key=str)
def decode_label(label):
    return sorted_labels[label]

class uc08(AbstractFunction):
    @property
    def name(self) -> str:
        return "uc08"

    def setup(self):
        self.predict_col = self.name
        model_file_name = f"../../../tpcxai_datasets/sf10/model/{self.name}/{self.name}.python.model"
        self.model = joblib.load(model_file_name)

    @forward(input_signatures=[PandasDataframe(columns=["scan_count", "scan_count_abs", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday", "dep0", "dep1", "dep2", "dep3", "dep4", "dep5", "dep6", "dep7", "dep8", "dep9", "dep10", "dep11", "dep12", "dep13", "dep14", "dep15", "dep16", "dep17", "dep18", "dep19", "dep20", "dep21", "dep22", "dep23", "dep24", "dep25", "dep26", "dep27", "dep28", "dep29", "dep30", "dep31", "dep32", "dep33", "dep34", "dep35", "dep36", "dep37", "dep38", "dep39", "dep40", "dep41", "dep42", "dep43", "dep44", "dep45", "dep46", "dep47", "dep48", "dep49", "dep50", "dep51", "dep52", "dep53", "dep54", "dep55", "dep56", "dep57", "dep58", "dep59", "dep60", "dep61", "dep62", "dep63", "dep64", "dep65", "dep66", "dep67"],
                                                column_types=[NdArrayType.INT32, NdArrayType.INT32, NdArrayType.INT32, NdArrayType.INT32, NdArrayType.INT32, NdArrayType.INT32, NdArrayType.INT32, NdArrayType.INT32, NdArrayType.INT32, NdArrayType.INT32, NdArrayType.INT32, NdArrayType.INT32, NdArrayType.INT32, NdArrayType.INT32, NdArrayType.INT32, NdArrayType.INT32, NdArrayType.INT32, NdArrayType.INT32, NdArrayType.INT32, NdArrayType.INT32, NdArrayType.INT32, NdArrayType.INT32, NdArrayType.INT32, NdArrayType.INT32, NdArrayType.INT32, NdArrayType.INT32, NdArrayType.INT32, NdArrayType.INT32, NdArrayType.INT32, NdArrayType.INT32, NdArrayType.INT32, NdArrayType.INT32, NdArrayType.INT32, NdArrayType.INT32, NdArrayType.INT32, NdArrayType.INT32, NdArrayType.INT32, NdArrayType.INT32, NdArrayType.INT32, NdArrayType.INT32, NdArrayType.INT32, NdArrayType.INT32, NdArrayType.INT32, NdArrayType.INT32, NdArrayType.INT32, NdArrayType.INT32, NdArrayType.INT32, NdArrayType.INT32, NdArrayType.INT32, NdArrayType.INT32, NdArrayType.INT32, NdArrayType.INT32, NdArrayType.INT32, NdArrayType.INT32, NdArrayType.INT32, NdArrayType.INT32, NdArrayType.INT32, NdArrayType.INT32, NdArrayType.INT32, NdArrayType.INT32, NdArrayType.INT32, NdArrayType.INT32, NdArrayType.INT32, NdArrayType.INT32, NdArrayType.INT32, NdArrayType.INT32, NdArrayType.INT32, NdArrayType.INT32, NdArrayType.INT32, NdArrayType.INT32, NdArrayType.INT32, NdArrayType.INT32, NdArrayType.INT32, NdArrayType.INT32, NdArrayType.INT32, NdArrayType.INT32, NdArrayType.INT32],                                                                                   column_shapes=[(None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, )])],
             output_signatures=[PandasDataframe(columns=["uc08"],
                                                column_types=[NdArrayType.INT64],
                                                column_shapes=[(None, )])])
    def forward(self, frames: pd.DataFrame) -> pd.DataFrame:
        print(len(frames))
        sparse_data = csr_matrix(frames.astype('int32'))
        predictions = self.model.predict(sparse_data)
        dec_fun = np.vectorize(decode_label)
        predict_df = pd.DataFrame({self.predict_col: dec_fun(predictions)})
        #predict_df.rename(columns={0:self.predict_col}, inplace=True)
        return predict_df
