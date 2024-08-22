import pickle

import duckdb
import numpy as np
from duckdb.typing import BIGINT, DOUBLE, FLOAT, VARCHAR, BOOLEAN
import pandas as pd
import time
import onnxruntime as ort

name = "pf2"

con = duckdb.connect("imbridge2.db")

start = time.perf_counter()
onnx_path = './test_raven/Expedia/expedia_LR_pipeline.onnx'
ortconfig = ort.SessionOptions()
expedia_onnx_session = ort.InferenceSession(onnx_path, sess_options=ortconfig)
expedia_label = expedia_onnx_session.get_outputs()[0]
numerical_columns = ['prop_location_score1', 'prop_location_score2', 'prop_log_historical_price', 'price_usd',
                         'orig_destination_distance', 'prop_review_score', 'avg_bookings_usd', 'stdev_bookings_usd']
categorical_columns = ['position', 'prop_country_id', 'prop_starrating', 'prop_brand_bool', 'count_clicks',
                           'count_bookings', 'year', 'month', 'weekofyear', 'time', 'site_id',
                           'visitor_location_country_id', 'srch_destination_id', 'srch_length_of_stay',
                           'srch_booking_window', 'srch_adults_count', 'srch_children_count', 'srch_room_count',
                           'srch_saturday_night_bool', 'random_bool']
expedia_input_columns = numerical_columns + categorical_columns
expedia_type_map = {
        'bool': np.int64,
        'int32': np.int64,
        'int64': np.int64,
        'float64': np.float32,
        'object': str,
}
stop = time.perf_counter()
with open(f"duck_{name}.log", 'a+') as f:
    f.write(f"{(stop-start)*1000}\n")

def udf(prop_location_score1, prop_location_score2, prop_log_historical_price, price_usd,
        orig_destination_distance, prop_review_score, avg_bookings_usd, stdev_bookings_usd,
        position_, prop_country_id, prop_starrating, prop_brand_bool, count_clicks, count_bookings,
        year_, month_, weekofyear_, time_, site_id, visitor_location_country_id, srch_destination_id,
        srch_length_of_stay, srch_booking_window, srch_adults_count, srch_children_count,
        srch_room_count, srch_saturday_night_bool, random_bool):

    def udf_wrap(*args):
        infer_batch = {
            elem: np.array(args[i]).astype(expedia_type_map[args[i].to_numpy().dtype.name]).reshape((-1, 1))
            for i, elem in enumerate(expedia_input_columns)
        }
        outputs = expedia_onnx_session.run([expedia_label.name], infer_batch)
        return outputs[0]

    return udf_wrap(prop_location_score1, prop_location_score2, prop_log_historical_price, price_usd,
        orig_destination_distance, prop_review_score, avg_bookings_usd, stdev_bookings_usd,
        position_, prop_country_id, prop_starrating, prop_brand_bool, count_clicks, count_bookings,
        year_, month_, weekofyear_, time_, site_id, visitor_location_country_id, srch_destination_id,
        srch_length_of_stay, srch_booking_window, srch_adults_count, srch_children_count,
        srch_room_count, srch_saturday_night_bool, random_bool)


con.create_function("udf", udf,
                    [DOUBLE, DOUBLE, DOUBLE, DOUBLE, DOUBLE, DOUBLE, DOUBLE, DOUBLE, VARCHAR, VARCHAR, BIGINT, BOOLEAN, BIGINT, BIGINT, VARCHAR, VARCHAR, VARCHAR, VARCHAR, VARCHAR, VARCHAR, VARCHAR, BIGINT, BIGINT,BIGINT, BIGINT, BIGINT, BOOLEAN, BOOLEAN], BIGINT, type="arrow")

#con.sql("SET threads TO 1;")

con.sql('''
explain analyze SELECT Expedia_S_listings_extension2.prop_id, Expedia_S_listings_extension2.srch_id, udf(prop_location_score1, prop_location_score2, prop_log_historical_price, price_usd,
                           orig_destination_distance, prop_review_score, avg_bookings_usd, stdev_bookings_usd,
                           position, prop_country_id, prop_starrating, prop_brand_bool, count_clicks, count_bookings,
                           year, month, weekofyear, time, site_id, visitor_location_country_id, srch_destination_id,
                           srch_length_of_stay, srch_booking_window, srch_adults_count, srch_children_count,
                           srch_room_count, srch_saturday_night_bool, random_bool)
FROM Expedia_S_listings_extension2 JOIN Expedia_R1_hotels2 ON Expedia_S_listings_extension2.prop_id = Expedia_R1_hotels2.prop_id
JOIN Expedia_R2_searches ON Expedia_S_listings_extension2.srch_id = Expedia_R2_searches.srch_id WHERE prop_location_score1 > 1 and prop_location_score2 > 0.1
and prop_log_historical_price > 4 and count_bookings > 5
and srch_booking_window > 10 and srch_length_of_stay > 1;
''')

print(name)