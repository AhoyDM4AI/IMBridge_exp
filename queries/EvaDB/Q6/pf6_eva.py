import pickle
import pandas as pd
import numpy as np
import onnxruntime as ort

from evadb.catalog.catalog_type import NdArrayType

from evadb.functions.abstract.abstract_function import AbstractFunction
from evadb.functions.decorators.decorators import setup, forward
from evadb.functions.decorators.io_descriptors.data_types import PandasDataframe



class pf6(AbstractFunction):
    @property
    def name(self) -> str:
        return "pf6"

    def setup(self):
        self.predict_col = self.name
        onnx_path = '"../../../raven_datasets/Hospital/hospital_mlp_pipeline.onnx' 
        ortconfig = ort.SessionOptions()
        self.onnx_session = ort.InferenceSession(onnx_path, sess_options=ortconfig)
        self.label = self.onnx_session.get_outputs()[0]
        numerical_columns = ['hematocrit', 'neutrophils', 'sodium', 'glucose', 'bloodureanitro', 'creatinine', 'bmi', 'pulse', 'respiration', 'secondarydiagnosisnonicd9']
        categorical_columns = ['rcount', 'gender', 'dialysisrenalendstage', 'asthma', 'irondef', 'pneum', 'substancedependence', 'psychologicaldisordermajor', 'depress', 'psychother', 'fibrosisandother', 'malnutrition', 'hemo']
        self.input_columns = numerical_columns + categorical_columns
        self.type_map = {
            "hematocrit": np.float32, 
            "neutrophils": np.float32, 
            "sodium": np.float32, 
            "glucose": np.float32, 
            "bloodureanitro": np.float32, 
            "creatinine": np.float32, 
            "bmi": np.float32, 
            "pulse": np.int64, 
            "respiration": np.float32, 
            "secondarydiagnosisnonicd9": np.int64, 
            "rcount": str, 
            "gender": str, 
            "dialysisrenalendstage": np.int64, 
            "asthma": np.int64, 
            "irondef": np.int64, 
            "pneum": np.int64, 
            "substancedependence": np.int64, 
            "psychologicaldisordermajor": np.int64, 
            "depress": np.int64, 
            "psychother": np.int64, 
            "fibrosisandother": np.int64, 
            "malnutrition": np.int64, 
            "hemo": np.int64
        }

    @forward(input_signatures=[PandasDataframe(columns=["hematocrit", "neutrophils", "sodium", "glucose", "bloodureanitro", "creatinine", "bmi", "pulse", "respiration", "secondarydiagnosisnonicd9", "rcount", "gender", "dialysisrenalendstage", "asthma", "irondef", "pneum", "substancedependence", "psychologicaldisordermajor", "depress", "psychother", "fibrosisandother", "malnutrition", "hemo"],
                                                column_types=[NdArrayType.FLOAT64, NdArrayType.FLOAT64, NdArrayType.FLOAT64, NdArrayType.FLOAT64, NdArrayType.FLOAT64, NdArrayType.FLOAT64, NdArrayType.FLOAT64, NdArrayType.INT64, NdArrayType.FLOAT64, NdArrayType.INT64, NdArrayType.STR, NdArrayType.STR, NdArrayType.INT64, NdArrayType.INT64, NdArrayType.INT64, NdArrayType.INT64, NdArrayType.INT64, NdArrayType.INT64, NdArrayType.INT64, NdArrayType.INT64, NdArrayType.INT64, NdArrayType.INT64, NdArrayType.INT64],
                                                column_shapes=[(None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, )])],
                                                output_signatures=[PandasDataframe(columns=["pf6"],
                                                column_types=[NdArrayType.INT64],
                                                column_shapes=[(None, )])])
    def forward(self, frames: pd.DataFrame) -> pd.DataFrame:
        print(len(frames))
        infer_batch = {
            elem: frames[elem].to_numpy().astype(self.type_map[elem]).reshape((-1, 1))
            for i, elem in enumerate(self.input_columns)
        }
        outputs = self.onnx_session.run([self.label.name], infer_batch)
        predictions = outputs[0]
        predict_df = pd.DataFrame({self.predict_col: predictions})
        return predict_df
