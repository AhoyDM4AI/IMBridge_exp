import pickle
import pandas as pd
import numpy as np
import onnxruntime as ort

from evadb.catalog.catalog_type import NdArrayType

from evadb.functions.abstract.abstract_function import AbstractFunction
from evadb.functions.decorators.decorators import setup, forward
from evadb.functions.decorators.io_descriptors.data_types import PandasDataframe



class pf4(AbstractFunction):
    @property
    def name(self) -> str:
        return "pf4"

    def setup(self):
        self.predict_col = self.name
        onnx_path = '"../../../raven_datasets/Flights/flights_rf_pipeline.onnx'
        ortconfig = ort.SessionOptions()
        self.onnx_session = ort.InferenceSession(onnx_path, sess_options=ortconfig)
        self.label = self.expedia_onnx_session.get_outputs()[0]
        numerical_columns = ['slatitude', 'slongitude', 'dlatitude', 'dlongitude']
        categorical_columns = ['name1', 'name2', 'name4', 'acountry', 'active',
	'scity', 'scountry', 'stimezone', 'sdst',
	'dcity', 'dcountry', 'dtimezone', 'ddst']
        self.input_columns = numerical_columns + categorical_columns
        self.type_map = {
            "slatitude": np.float32,
            "slongitude": np.float32, 
            "dlatitude": np.float32, 
            "dlongitude": np.float32, 
            "name1": np.int64, 
            "name2": str, 
            "name4": str, 
            "acountry": str, 
            "active": str, 
            "scity": str, 
            "scountry": str, 
            "stimezone": np.int64, 
            "sdst": str, 
            "dcity": str, 
            "dcountry": str, 
            "dtimezone": np.int64, 
            "ddst": str
        }

    @forward(input_signatures=[PandasDataframe(columns=["slatitude", "slongitude", "dlatitude", "dlongitude", "name1", "name2", "name4", "acountry", "active", "scity", "scountry", "stimezone", "sdst", "dcity", "dcountry", "dtimezone", "ddst"],
                                               column_types=[NdArrayType.FLOAT64, NdArrayType.FLOAT64, NdArrayType.FLOAT64, NdArrayType.FLOAT64, NdArrayType.INT64, NdArrayType.STR, NdArrayType.STR, NdArrayType.STR, NdArrayType.STR, NdArrayType.STR, NdArrayType.STR, NdArrayType.INT64, NdArrayType.STR, NdArrayType.STR, NdArrayType.STR, NdArrayType.INT64, NdArrayType.STR],
                                               column_shapes=[(None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, )])],
             output_signatures=[PandasDataframe(columns=["pf4"],
                                                column_types=[NdArrayType.INT64],
                                                column_shapes=[(None, )])])
    def forward(self, frames: pd.DataFrame) -> pd.DataFrame:
        infer_batch = {
            elem: frames[elem].to_numpy().astype(self.type_map[elem]).reshape((-1, 1))
            for i, elem in enumerate(self.input_columns)
        }
        outputs = self.onnx_session.run([self.label.name], infer_batch)
        predictions = outputs[0]
        print(len(frames))
        predict_df = pd.DataFrame({self.predict_col: predictions})
        return predict_df
