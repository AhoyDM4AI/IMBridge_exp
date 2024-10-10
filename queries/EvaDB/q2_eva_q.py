import evadb

cursor = evadb.connect().cursor()
name = "pf1"

params = {
    "user": "root",
    "host": "49.52.27.23",
    "port": "4001",
    "database": "raven",
    "password": "oGslD19GXXy6F5bhzzox",
    #"connection_timeout": "36000000000"
}
query = f"CREATE DATABASE backend_data WITH ENGINE='mysql', PARAMETERS={params};"
cursor.query(query).df()

cursor.query(f"CREATE FUNCTION IF NOT EXISTS {name} IMPL './{name}_eva.py'").execute()

cursor.query("use backend_data {drop view if exists temp_eva_pf1}").execute()

#cursor.query("use backend_data {drop table if exists temp_eva_pf1}").execute()

#cursor.query("use backend_data {create table temp_eva_pf1(srch_id VARCHAR(32), prop_id VARCHAR(32), prop_location_score1 DOUBLE, prop_location_score2 DOUBLE, prop_log_historical_price DOUBLE, price_usd DOUBLE, orig_destination_distance DOUBLE, prop_review_score DOUBLE, avg_bookings_usd DOUBLE, stdev_bookings_usd DOUBLE, position VARCHAR(32), prop_country_id VARCHAR(32), prop_starrating INTEGER, prop_brand_bool INTEGER, count_clicks INTEGER, count_bookings INTEGER, year VARCHAR(32), month VARCHAR(32), weekofyear VARCHAR(32), time VARCHAR(32),  site_id VARCHAR(32), visitor_location_country_id VARCHAR(32), srch_destination_id VARCHAR(32), srch_length_of_stay INTEGER, srch_booking_window INTEGER, srch_adults_count INTEGER, srch_children_count INTEGER, srch_room_count INTEGER, srch_saturday_night_bool INTEGER, random_bool INTEGER)}").execute()

#cursor.query("use backend_data {INSERT INTO temp_eva_pf1 SELECT Expedia_S_listings.srch_id, Expedia_S_listings.prop_id, prop_location_score1, prop_location_score2, prop_log_historical_price, price_usd, orig_destination_distance, prop_review_score, avg_bookings_usd, stdev_bookings_usd,  position, prop_country_id, prop_starrating, CAST(prop_brand_bool AS SIGNED), count_clicks,  count_bookings, year, month, weekofyear, time, site_id, visitor_location_country_id, srch_destination_id, srch_length_of_stay, srch_booking_window, srch_adults_count,  srch_children_count, srch_room_count, CAST(srch_saturday_night_bool AS SIGNED), CAST(random_bool AS SIGNED) FROM Expedia_S_listings JOIN Expedia_R1_hotels ON Expedia_S_listings.prop_id = Expedia_R1_hotels.prop_id JOIN Expedia_R2_searches ON Expedia_S_listings.srch_id = Expedia_R2_searches.srch_id}").execute()

cursor.query("use backend_data {CREATE VIEW temp_eva_pf1 AS SELECT Expedia_S_listings_eva.srch_id, Expedia_S_listings_eva.prop_id, prop_location_score1, prop_location_score2, prop_log_historical_price, price_usd, orig_destination_distance, prop_review_score, avg_bookings_usd, stdev_bookings_usd,  position, prop_country_id, prop_starrating, prop_brand_bool, count_clicks,  count_bookings, year, month, weekofyear, time, site_id, visitor_location_country_id, srch_destination_id, srch_length_of_stay, srch_booking_window, srch_adults_count,  srch_children_count, srch_room_count, srch_saturday_night_bool, random_bool FROM Expedia_S_listings_eva JOIN Expedia_R1_hotels_eva ON Expedia_S_listings_eva.prop_id = Expedia_R1_hotels_eva.prop_id JOIN Expedia_R2_searches_eva ON Expedia_S_listings_eva.srch_id = Expedia_R2_searches_eva.srch_id}").execute()

#cursor.query("use backend_data {CREATE TABLE temp_eva_pf1 AS SELECT Expedia_S_listings.srch_id, Expedia_S_listings.prop_id FROM Expedia_S_listings JOIN Expedia_R1_hotels ON Expedia_S_listings.prop_id = Expedia_R1_hotels.prop_id JOIN Expedia_R2_searches ON Expedia_S_listings.srch_id = Expedia_R2_searches.srch_id WHERE prop_location_score1 > 1 and prop_location_score2 > 0.1 and prop_log_historical_price > 4 and count_bookings > 5 and srch_booking_window > 10 and srch_length_of_stay > 1}").execute()

#cursor.query(f"create table tb_eva_{name} as select transactionID, amount_norm, business_hour_norm from backend_data.temp_eva_uc10;").execute()

#print(cursor.query("select srch_id, prop_id from backend_data.temp_eva_pf1;").df())
print(cursor.query("select srch_id, prop_id, pf1(prop_location_score1, prop_location_score2, prop_log_historical_price, price_usd, orig_destination_distance, prop_review_score, avg_bookings_usd, stdev_bookings_usd,  position, prop_country_id, prop_starrating, prop_brand_bool, count_clicks,  count_bookings, year, month, weekofyear, time, site_id, visitor_location_country_id, srch_destination_id, srch_length_of_stay, srch_booking_window, srch_adults_count,  srch_children_count, srch_room_count, srch_saturday_night_bool, random_bool) from backend_data.temp_eva_pf1 WHERE prop_location_score1 > 1 and prop_location_score2 > 0.1 and prop_log_historical_price > 4 and count_bookings > 5 and srch_booking_window > 10 and srch_length_of_stay > 1;").df())
