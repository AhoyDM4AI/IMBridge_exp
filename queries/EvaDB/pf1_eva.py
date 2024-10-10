import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from evadb.catalog.catalog_type import NdArrayType

from evadb.functions.abstract.abstract_function import AbstractFunction
from evadb.functions.decorators.decorators import setup, forward
from evadb.functions.decorators.io_descriptors.data_types import PandasDataframe



class pf1(AbstractFunction):
    @property
    def name(self) -> str:
        return "pf1"

    def setup(self):
        self.predict_col = self.name
        scaler_path = "/home/test_raven/Expedia/expedia_standard_scale_model.pkl"
        enc_path = "/home/test_raven/Expedia/expedia_one_hot_encoder.pkl"
        model_path = "/home/test_raven/Expedia/expedia_dt_model.pkl"
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        with open(enc_path, 'rb') as f:
            self.enc = pickle.load(f)
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

    @forward(input_signatures=[PandasDataframe(columns=["prop_location_score1", "prop_location_score2", "prop_log_historical_price", "price_usd", "orig_destination_distance", "prop_review_score", "avg_bookings_usd", "stdev_bookings_usd", "position_", "prop_country_id", "prop_starrating", "prop_brand_bool", "count_clicks", "count_bookings", "year_", "month_", "weekofyear_", "time_", "site_id", "visitor_location_country_id", "srch_destination_id", "srch_length_of_stay", "srch_booking_window", "srch_adults_count", "srch_children_count", "srch_room_count", "srch_saturday_night_bool", "random_bool"],
                                               column_types=[NdArrayType.FLOAT64, NdArrayType.FLOAT64, NdArrayType.FLOAT64, NdArrayType.FLOAT64, NdArrayType.FLOAT64, NdArrayType.FLOAT64, NdArrayType.FLOAT64, NdArrayType.FLOAT64, NdArrayType.STR, NdArrayType.STR, NdArrayType.INT64, NdArrayType.INT64, NdArrayType.INT64, NdArrayType.INT64, NdArrayType.STR, NdArrayType.STR, NdArrayType.STR, NdArrayType.STR, NdArrayType.STR, NdArrayType.STR, NdArrayType.STR, NdArrayType.INT64, NdArrayType.INT64, NdArrayType.INT64, NdArrayType.INT64, NdArrayType.INT64, NdArrayType.INT64, NdArrayType.INT64],
                                               column_shapes=[(None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, ), (None, )])],
             output_signatures=[PandasDataframe(columns=["pf1"],
                                                column_types=[NdArrayType.INT64],
                                                column_shapes=[(None, )])])
    def forward(self, frames: pd.DataFrame) -> pd.DataFrame:
        data = np.split(frames.to_numpy(), np.array([8]), axis = 1)
        numerical = data[0]
        categorical = data[1]
        X = np.hstack((self.scaler.transform(numerical), self.enc.transform(categorical).toarray()))
        predictions = self.model.predict(X)
        print(len(frames))
        predict_df = pd.DataFrame({self.predict_col: predictions})
        return predict_df
