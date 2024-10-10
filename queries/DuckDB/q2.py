import pickle

import duckdb
import numpy as np
from duckdb.typing import BIGINT, DOUBLE, FLOAT, VARCHAR, BOOLEAN
import pandas as pd
import time
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

name = "q2"

con = duckdb.connect("imbridge2.db")

start = time.perf_counter()
scaler_path = '/home/test_raven/Expedia/expedia_standard_scale_model.pkl'
enc_path = '/home/test_raven/Expedia/expedia_one_hot_encoder.pkl'
model_path = '/home/test_raven/Expedia/expedia_dt_model.pkl'
with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)
with open(enc_path, 'rb') as f:
    enc = pickle.load(f)
with open(model_path, 'rb') as f:
    model = pickle.load(f)
stop = time.perf_counter()
with open(f"duck_{name}.log", 'a+') as f:
        f.write(f"{(stop-start)*1000}\n")

def udf(prop_location_score1, prop_location_score2, prop_log_historical_price, price_usd,
        orig_destination_distance, prop_review_score, avg_bookings_usd, stdev_bookings_usd,
        position_, prop_country_id, prop_starrating, prop_brand_bool, count_clicks, count_bookings,
        year_, month_, weekofyear_, time_, site_id, visitor_location_country_id, srch_destination_id,
        srch_length_of_stay, srch_booking_window, srch_adults_count, srch_children_count,
        srch_room_count, srch_saturday_night_bool, random_bool):

    data = np.column_stack([prop_location_score1, prop_location_score2, prop_log_historical_price, price_usd,
                            orig_destination_distance, prop_review_score, avg_bookings_usd, stdev_bookings_usd,
                            position_, prop_country_id, prop_starrating, prop_brand_bool, count_clicks, count_bookings,
                            year_, month_, weekofyear_, time_, site_id, visitor_location_country_id,
                            srch_destination_id,
                            srch_length_of_stay, srch_booking_window, srch_adults_count, srch_children_count,
                            srch_room_count, srch_saturday_night_bool, random_bool])
    data = np.split(data, np.array([8]), axis=1)
    numerical = data[0]
    categorical = data[1]

    X = np.hstack((scaler.transform(numerical), enc.transform(categorical).toarray()))
    return model.predict(X)


con.create_function("udf", udf,
                    [DOUBLE, DOUBLE, DOUBLE, DOUBLE, DOUBLE, DOUBLE, DOUBLE, DOUBLE, VARCHAR, VARCHAR, BIGINT, BOOLEAN, BIGINT, BIGINT, VARCHAR, VARCHAR, VARCHAR, VARCHAR, VARCHAR, VARCHAR, VARCHAR, BIGINT, BIGINT,BIGINT, BIGINT, BIGINT, BOOLEAN, BOOLEAN], BIGINT, type="arrow")


# con.sql("SET threads TO 1;")

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
