import pickle
import pandas as pd
import numpy as np
import onnxruntime as ort

from evadb.catalog.catalog_type import NdArrayType

from evadb.functions.abstract.abstract_function import AbstractFunction
from evadb.functions.decorators.decorators import setup, forward
from evadb.functions.decorators.io_descriptors.data_types import PandasDataframe



class pf2(AbstractFunction):
    @property
    def name(self) -> str:
        return "pf2"

    def setup(self):
        self.predict_col = self.name
        onnx_path = '/home/test_raven/Expedia/expedia_dt_pipeline.onnx'
        ortconfig = ort.SessionOptions()
        self.expedia_onnx_session = ort.InferenceSession(onnx_path, sess_options=ortconfig)
        self.expedia_label = self.expedia_onnx_session.get_outputs()[0]
        numerical_columns = ['prop_location_score1', 'prop_location_score2', 'prop_log_historical_price', 'price_usd', 'orig_destination_distance', 'prop_review_score', 'avg_bookings_usd', 'stdev_bookings_usd']
        categorical_columns = ['position', 'prop_country_id', 'prop_starrating', 'prop_brand_bool', 'count_clicks','count_bookings', 'year', 'month', 'weekofyear', 'time', 'site_id', 'visitor_location_country_id', 'srch_destination_id', 'srch_length_of_stay', 'srch_booking_window', 'srch_adults_count', 'srch_children_count', 'srch_room_count', 'srch_saturday_night_bool', 'random_bool']
        self.expedia_input_columns = numerical_columns + categorical_columns
        self.expedia_type_map = {
            'prop_location_score1': np.float32,
            'prop_location_score2': np.float32,
            'prop_log_historical_price': np.float32,
            'price_usd': np.float32,
            'orig_destination_distance': np.float32,
            'prop_review_score': np.float32,
            'avg_bookings_usd': np.float32,
            'stdev_bookings_usd': np.float32,
            'position': str,
            'prop_country_id': str,
            'prop_starrating': np.int64,
            'prop_brand_bool': np.int64,
            'count_clicks': np.int64,
            'count_bookings': np.int64,
            'year': str,
            'month': str,
            'weekofyear': str,
            'time': str,
            'site_id': str,
            'visitor_location_country_id': str,
            'srch_destination_id': str,
            'srch_length_of_stay': np.int64,
            'srch_booking_window': np.int64,
            'srch_adults_count': np.int64,
            'srch_children_count': np.int64,
            'srch_room_count': np.int64,
            'srch_saturday_night_bool': np.int64,
            'random_bool': np.int64
        }

    @forward(input_signatures=[PandasDataframe(columns=["prop_location_score1", "prop_location_score2", "prop_log_historical_price", "price_usd", "orig_destination_distance", "prop_review_score", "avg_bookings_usd", "stdev_bookings_usd", "position", "prop_country_id", "prop_starrating", "prop_brand_bool", "count_clicks", "count_bookings", "year", "month", "weekofyear", "time", "site_id", "visitor_location_country_id", "srch_destination_id", "srch_length_of_stay", "srch_booking_window", "srch_adults_count", "srch_children_count", "srch_room_count", "srch_saturday_night_bool", "random_bool"],
                                               column_types=[NdArrayType.FLOAT32, NdArrayType.FLOAT32, NdArrayType.FLOAT32, NdArrayType.FLOAT32, NdArrayType.FLOAT32, NdArrayType.FLOAT32, NdArrayType.FLOAT32, NdArrayType.FLOAT32, NdArrayType.STR, NdArrayType.STR, NdArrayType.INT64, NdArrayType.INT64, NdArrayType.INT64, NdArrayType.INT64, NdArrayType.STR, NdArrayType.STR, NdArrayType.STR, NdArrayType.STR, NdArrayType.STR, NdArrayType.STR, NdArrayType.STR, NdArrayType.INT64, NdArrayType.INT64, NdArrayType.INT64, NdArrayType.INT64, NdArrayType.INT64, NdArrayType.INT64, NdArrayType.INT64],
                                               column_shapes=[(None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, )])],
             output_signatures=[PandasDataframe(columns=["pf2"],
                                                column_types=[NdArrayType.INT64],
                                                column_shapes=[(None, )])])
    def forward(self, frames: pd.DataFrame) -> pd.DataFrame:
        infer_batch = {
            elem: frames[elem].to_numpy().astype(self.expedia_type_map[elem]).reshape((-1, 1))
            for i, elem in enumerate(self.expedia_input_columns)
        }

        outputs = self.expedia_onnx_session.run([self.expedia_label.name], infer_batch)
        predictions = outputs[0]
        print(len(frames))
        predict_df = pd.DataFrame({self.predict_col: predictions})
        return predict_df
