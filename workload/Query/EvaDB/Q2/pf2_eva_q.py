import evadb

cursor = evadb.connect().cursor()
name = "pf2"

params = {
    "user": "root",
    "host": "xxx",
    "port": "2881",
    "database": "raven",
    "password": "oGslD19GXXy6F5bhzzox",
    #"connection_timeout": "36000000000"
}
query = f"CREATE DATABASE backend_data WITH ENGINE='mysql', PARAMETERS={params};"
cursor.query(query).df()

cursor.query(f"CREATE FUNCTION IF NOT EXISTS {name} IMPL './{name}_eva.py'").execute()

cursor.query("use backend_data {drop view if exists temp_eva_pf2}").execute()

cursor.query("use backend_data {CREATE VIEW temp_eva_pf2 AS SELECT Expedia_S_listings_eva.srch_id, Expedia_S_listings_eva.prop_id, prop_location_score1, prop_location_score2, prop_log_historical_price, price_usd, orig_destination_distance, prop_review_score, avg_bookings_usd, stdev_bookings_usd,  position, prop_country_id, prop_starrating, prop_brand_bool, count_clicks,  count_bookings, year, month, weekofyear, time, site_id, visitor_location_country_id, srch_destination_id, srch_length_of_stay, srch_booking_window, srch_adults_count,  srch_children_count, srch_room_count, srch_saturday_night_bool, random_bool FROM Expedia_S_listings_eva JOIN Expedia_R1_hotels_eva ON Expedia_S_listings_eva.prop_id = Expedia_R1_hotels_eva.prop_id JOIN Expedia_R2_searches_eva ON Expedia_S_listings_eva.srch_id = Expedia_R2_searches_eva.srch_id}").execute()

print(cursor.query("select srch_id, prop_id, pf2(prop_location_score1, prop_location_score2, prop_log_historical_price, price_usd, orig_destination_distance, prop_review_score, avg_bookings_usd, stdev_bookings_usd,  position, prop_country_id, prop_starrating, prop_brand_bool, count_clicks,  count_bookings, year, month, weekofyear, time, site_id, visitor_location_country_id, srch_destination_id, srch_length_of_stay, srch_booking_window, srch_adults_count,  srch_children_count, srch_room_count, srch_saturday_night_bool, random_bool) from backend_data.temp_eva_pf1 WHERE prop_location_score1 > 1 and prop_location_score2 > 0.1 and prop_log_historical_price > 4 and count_bookings > 5 and srch_booking_window > 10 and srch_length_of_stay > 1;").df())
