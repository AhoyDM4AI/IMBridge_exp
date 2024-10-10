import pymysql
import json
import time
import pandas as pd
import os
import time

trace_on = "SET ob_enable_show_trace = 1;"
show_trace = "SHOW TRACE;"
plan_flush = "ALTER SYSTEM FLUSH PLAN CACHE;"
max_rows = 2048

path = "./stat.csv"

# bs_stars = [
# 	3072,	2816,	1792,	3072,	2560,	4352,	3072,	512,	2048,	1536,	3072,	6656
# ]

bs_stars = [
	3072,	4352,	1536,	6656
]

repeat = 1
count = 1

show_result = False

def run_sql(cur, sql):
	with open("/home/test/experiments/oceanbase/opt/both_pps.log", "a+") as log:
		log.write('\n' + sql + '\n')
	cur.execute(plan_flush)
	# mr = bs_stars[count -1]
	mr = 256
	max_rows_set = f'ALTER SYSTEM SET _rowsets_max_rows = {mr}'
	cur.execute(max_rows_set)
	start = time.perf_counter()
	cur.execute(sql)
	stop = time.perf_counter()
	if (show_result):
		res = cur.fetchall()
		with open('./output.txt', 'w') as file:
			for row in res:
				#print(row)
				formatted_row = ' '.join(map(str, row)) + '\n'
				file.write(formatted_row)
	# time_consuming = analysis_trace(cur)
	time_consuming = stop-start
	print(time_consuming)
	return time_consuming

def analysis_trace(cur):
	cur.execute(show_trace)
	trace = cur.fetchone()
	if trace is not None:
		return trace[2]
	else:
		return -1

# raven predict sql
# select rate: 10%
expedia_sklearn_dt = '''SELECT /*+PARALLEL({0})*/ Expedia_S_listings{1}.srch_id, PREDICT expedia_sklearn_dt_{2}( 
prop_location_score1, prop_location_score2, prop_log_historical_price, price_usd, orig_destination_distance, 
prop_review_score, avg_bookings_usd, stdev_bookings_usd,  position, prop_country_id, prop_starrating, 
prop_brand_bool, count_clicks,  count_bookings, year, month, weekofyear, time, site_id, 
visitor_location_country_id, srch_destination_id, srch_length_of_stay, srch_booking_window, 
srch_adults_count,  srch_children_count, srch_room_count, srch_saturday_night_bool, random_bool) AS promotion_flag 
FROM Expedia_S_listings{1} JOIN Expedia_R1_hotels ON Expedia_S_listings{1}.prop_id = Expedia_R1_hotels.prop_id 
JOIN Expedia_R2_searches ON Expedia_S_listings{1}.srch_id = Expedia_R2_searches.srch_id 
WHERE prop_location_score1 > 1 and prop_location_score2 > 0.1 and prop_log_historical_price > 4 
and count_bookings > 5 and srch_booking_window > 4 and srch_length_of_stay > 1;'''

expedia_sklearn_dt_1 = '''SELECT /*+PARALLEL({0})*/ Expedia_S_listings{1}.srch_id, PREDICT expedia_sklearn_dt_{2}( 
prop_location_score1, prop_location_score2, prop_log_historical_price, price_usd, orig_destination_distance, 
prop_review_score, avg_bookings_usd, stdev_bookings_usd,  position, prop_country_id, prop_starrating, 
prop_brand_bool, count_clicks,  count_bookings, year, month, weekofyear, time, site_id, 
visitor_location_country_id, srch_destination_id, srch_length_of_stay, srch_booking_window, 
srch_adults_count,  srch_children_count, srch_room_count, srch_saturday_night_bool, random_bool) AS promotion_flag 
FROM Expedia_S_listings{1} JOIN Expedia_R1_hotels ON Expedia_S_listings{1}.prop_id = Expedia_R1_hotels.prop_id 
JOIN Expedia_R2_searches ON Expedia_S_listings{1}.srch_id = Expedia_R2_searches.srch_id 
WHERE prop_location_score1 > 1 and prop_location_score2 > 0.1 and prop_log_historical_price > 4 and count_bookings > 5 and srch_booking_window > 4 and srch_length_of_stay > 5;'''

expedia_sklearn_dt_10 = '''SELECT /*+PARALLEL({0})*/ Expedia_S_listings{1}.srch_id, PREDICT expedia_sklearn_dt_{2}( 
prop_location_score1, prop_location_score2, prop_log_historical_price, price_usd, orig_destination_distance, 
prop_review_score, avg_bookings_usd, stdev_bookings_usd,  position, prop_country_id, prop_starrating, 
prop_brand_bool, count_clicks,  count_bookings, year, month, weekofyear, time, site_id, 
visitor_location_country_id, srch_destination_id, srch_length_of_stay, srch_booking_window, 
srch_adults_count,  srch_children_count, srch_room_count, srch_saturday_night_bool, random_bool) AS promotion_flag 
FROM Expedia_S_listings{1} JOIN Expedia_R1_hotels ON Expedia_S_listings{1}.prop_id = Expedia_R1_hotels.prop_id 
JOIN Expedia_R2_searches ON Expedia_S_listings{1}.srch_id = Expedia_R2_searches.srch_id 
WHERE prop_location_score1 > 1 and prop_location_score2 > 0.1 and prop_log_historical_price > 4 
and count_bookings > 5 and srch_booking_window > 4 and srch_length_of_stay > 1;'''

expedia_sklearn_dt_30 = '''SELECT /*+PARALLEL({0})*/ Expedia_S_listings{1}.srch_id, PREDICT expedia_sklearn_dt_{2}( 
prop_location_score1, prop_location_score2, prop_log_historical_price, price_usd, orig_destination_distance, 
prop_review_score, avg_bookings_usd, stdev_bookings_usd,  position, prop_country_id, prop_starrating, 
prop_brand_bool, count_clicks,  count_bookings, year, month, weekofyear, time, site_id, 
visitor_location_country_id, srch_destination_id, srch_length_of_stay, srch_booking_window, 
srch_adults_count,  srch_children_count, srch_room_count, srch_saturday_night_bool, random_bool) AS promotion_flag 
FROM Expedia_S_listings{1} JOIN Expedia_R1_hotels ON Expedia_S_listings{1}.prop_id = Expedia_R1_hotels.prop_id 
JOIN Expedia_R2_searches ON Expedia_S_listings{1}.srch_id = Expedia_R2_searches.srch_id 
WHERE prop_location_score1 > 1.5 and prop_location_score2 > 0.1;'''

expedia_sklearn_dt_60 = '''SELECT /*+PARALLEL({0})*/ Expedia_S_listings{1}.srch_id, PREDICT expedia_sklearn_dt_{2}( 
prop_location_score1, prop_location_score2, prop_log_historical_price, price_usd, orig_destination_distance, 
prop_review_score, avg_bookings_usd, stdev_bookings_usd,  position, prop_country_id, prop_starrating, 
prop_brand_bool, count_clicks,  count_bookings, year, month, weekofyear, time, site_id, 
visitor_location_country_id, srch_destination_id, srch_length_of_stay, srch_booking_window, 
srch_adults_count,  srch_children_count, srch_room_count, srch_saturday_night_bool, random_bool) AS promotion_flag 
FROM Expedia_S_listings{1} JOIN Expedia_R1_hotels ON Expedia_S_listings{1}.prop_id = Expedia_R1_hotels.prop_id 
JOIN Expedia_R2_searches ON Expedia_S_listings{1}.srch_id = Expedia_R2_searches.srch_id 
WHERE count_bookings > 5 and srch_booking_window > 1;'''

expedia_sklearn_dt_100 = '''SELECT /*+PARALLEL({0})*/ Expedia_S_listings{1}.srch_id, PREDICT expedia_sklearn_dt_{2}( 
prop_location_score1, prop_location_score2, prop_log_historical_price, price_usd, orig_destination_distance, 
prop_review_score, avg_bookings_usd, stdev_bookings_usd,  position, prop_country_id, prop_starrating, 
prop_brand_bool, count_clicks,  count_bookings, year, month, weekofyear, time, site_id, 
visitor_location_country_id, srch_destination_id, srch_length_of_stay, srch_booking_window, 
srch_adults_count,  srch_children_count, srch_room_count, srch_saturday_night_bool, random_bool) AS promotion_flag 
FROM Expedia_S_listings{1} JOIN Expedia_R1_hotels ON Expedia_S_listings{1}.prop_id = Expedia_R1_hotels.prop_id 
JOIN Expedia_R2_searches ON Expedia_S_listings{1}.srch_id = Expedia_R2_searches.srch_id;'''

'''
SELECT /*+PARALLEL(1)*/ Expedia_S_listings_1G.srch_id, PREDICT expedia_sklearn_dt_uninit( 
prop_location_score1, prop_location_score2, prop_log_historical_price, price_usd, orig_destination_distance, 
prop_review_score, avg_bookings_usd, stdev_bookings_usd,  position, prop_country_id, prop_starrating, 
prop_brand_bool, count_clicks,  count_bookings, year, month, weekofyear, time, site_id, 
visitor_location_country_id, srch_destination_id, srch_length_of_stay, srch_booking_window, 
srch_adults_count,  srch_children_count, srch_room_count, srch_saturday_night_bool, random_bool) AS promotion_flag 
FROM Expedia_S_listings_1G JOIN Expedia_R1_hotels ON Expedia_S_listings_1G.prop_id = Expedia_R1_hotels.prop_id 
JOIN Expedia_R2_searches ON Expedia_S_listings_1G.srch_id = Expedia_R2_searches.srch_id 
WHERE prop_location_score1 > 1 and prop_location_score2 > 0.1 and prop_log_historical_price > 4 
and count_bookings > 5 and srch_booking_window > 4 and srch_length_of_stay > 1 limit 5000;

SELECT /*+PARALLEL(1)*/ Expedia_S_listings_1G.srch_id, PREDICT expedia_sklearn_dt_uninit( 
prop_location_score1, prop_location_score2, prop_log_historical_price, price_usd, orig_destination_distance, 
prop_review_score, avg_bookings_usd, stdev_bookings_usd,  position, prop_country_id, prop_starrating, 
prop_brand_bool, count_clicks,  count_bookings, year, month, weekofyear, time, site_id, 
visitor_location_country_id, srch_destination_id, srch_length_of_stay, srch_booking_window, 
srch_adults_count,  srch_children_count, srch_room_count, srch_saturday_night_bool, random_bool) AS promotion_flag 
FROM Expedia_S_listings_1G JOIN Expedia_R1_hotels ON Expedia_S_listings_1G.prop_id = Expedia_R1_hotels.prop_id 
JOIN Expedia_R2_searches ON Expedia_S_listings_1G.srch_id = Expedia_R2_searches.srch_id 
WHERE prop_location_score1 > 1 and prop_location_score2 > 0.1 and prop_log_historical_price > 4 
and count_bookings > 5 and srch_booking_window > 4 and srch_length_of_stay > 1 limit 10000;

SELECT /*+PARALLEL(1)*/ Expedia_S_listings_1G.srch_id, PREDICT expedia_sklearn_dt_uninit( 
prop_location_score1, prop_location_score2, prop_log_historical_price, price_usd, orig_destination_distance, 
prop_review_score, avg_bookings_usd, stdev_bookings_usd,  position, prop_country_id, prop_starrating, 
prop_brand_bool, count_clicks,  count_bookings, year, month, weekofyear, time, site_id, 
visitor_location_country_id, srch_destination_id, srch_length_of_stay, srch_booking_window, 
srch_adults_count,  srch_children_count, srch_room_count, srch_saturday_night_bool, random_bool) AS promotion_flag 
FROM Expedia_S_listings_1G JOIN Expedia_R1_hotels ON Expedia_S_listings_1G.prop_id = Expedia_R1_hotels.prop_id 
JOIN Expedia_R2_searches ON Expedia_S_listings_1G.srch_id = Expedia_R2_searches.srch_id 
WHERE prop_location_score1 > 1 and prop_location_score2 > 0.1 and prop_log_historical_price > 4 
and count_bookings > 5 and srch_booking_window > 4 and srch_length_of_stay > 1 limit 50000;

SELECT /*+PARALLEL(1)*/ Expedia_S_listings_1G.srch_id, PREDICT expedia_sklearn_dt_uninit( 
prop_location_score1, prop_location_score2, prop_log_historical_price, price_usd, orig_destination_distance, 
prop_review_score, avg_bookings_usd, stdev_bookings_usd,  position, prop_country_id, prop_starrating, 
prop_brand_bool, count_clicks,  count_bookings, year, month, weekofyear, time, site_id, 
visitor_location_country_id, srch_destination_id, srch_length_of_stay, srch_booking_window, 
srch_adults_count,  srch_children_count, srch_room_count, srch_saturday_night_bool, random_bool) AS promotion_flag 
FROM Expedia_S_listings_1G JOIN Expedia_R1_hotels ON Expedia_S_listings_1G.prop_id = Expedia_R1_hotels.prop_id 
JOIN Expedia_R2_searches ON Expedia_S_listings_1G.srch_id = Expedia_R2_searches.srch_id 
WHERE prop_location_score1 > 1 and prop_location_score2 > 0.1 and prop_log_historical_price > 4 
and count_bookings > 5 and srch_booking_window > 4 and srch_length_of_stay > 1 limit 100000;

SELECT /*+PARALLEL(1)*/ Expedia_S_listings_1G.srch_id, PREDICT expedia_sklearn_dt_uninit( 
prop_location_score1, prop_location_score2, prop_log_historical_price, price_usd, orig_destination_distance, 
prop_review_score, avg_bookings_usd, stdev_bookings_usd,  position, prop_country_id, prop_starrating, 
prop_brand_bool, count_clicks,  count_bookings, year, month, weekofyear, time, site_id, 
visitor_location_country_id, srch_destination_id, srch_length_of_stay, srch_booking_window, 
srch_adults_count,  srch_children_count, srch_room_count, srch_saturday_night_bool, random_bool) AS promotion_flag 
FROM Expedia_S_listings_1G JOIN Expedia_R1_hotels ON Expedia_S_listings_1G.prop_id = Expedia_R1_hotels.prop_id 
JOIN Expedia_R2_searches ON Expedia_S_listings_1G.srch_id = Expedia_R2_searches.srch_id 
WHERE prop_location_score1 > 1 and prop_location_score2 > 0.1 and prop_log_historical_price > 4 
and count_bookings > 5 and srch_booking_window > 4 and srch_length_of_stay > 1 limit 500000;
'''

expedia_onnx_dt = '''SELECT /*+PARALLEL({0})*/ Expedia_S_listings{1}.srch_id, PREDICT expedia_onnx_dt_{2}( 
prop_location_score1, prop_location_score2, prop_log_historical_price, price_usd, orig_destination_distance, 
prop_review_score, avg_bookings_usd, stdev_bookings_usd,  position, prop_country_id, prop_starrating, 
prop_brand_bool, count_clicks,  count_bookings, year, month, weekofyear, time, site_id, 
visitor_location_country_id, srch_destination_id, srch_length_of_stay, srch_booking_window, 
srch_adults_count,  srch_children_count, srch_room_count, srch_saturday_night_bool, random_bool) AS promotion_flag 
FROM Expedia_S_listings{1} JOIN Expedia_R1_hotels ON Expedia_S_listings{1}.prop_id = Expedia_R1_hotels.prop_id 
JOIN Expedia_R2_searches ON Expedia_S_listings{1}.srch_id = Expedia_R2_searches.srch_id 
WHERE prop_location_score1 > 1 and prop_location_score2 > 0.1 and prop_log_historical_price > 4 
and count_bookings > 5 and srch_booking_window > 4 and srch_length_of_stay > 1;'''

flights_sklearn_rf = '''SELECT /*+PARALLEL({0})*/ Flights_S_routes{1}.airlineid, PREDICT flights_sklearn_rf_{2}( 
slatitude, slongitude, dlatitude, dlongitude, name1, name2, name4, acountry, active, 
scity, scountry, stimezone, sdst, dcity, dcountry, dtimezone, ddst) AS codeshare FROM 
Flights_S_routes{1} JOIN Flights_R1_airlines ON Flights_S_routes{1}.airlineid = Flights_R1_airlines.airlineid 
JOIN Flights_R2_sairports ON Flights_S_routes{1}.sairportid = Flights_R2_sairports.sairportid 
JOIN Flights_R3_dairports ON Flights_S_routes{1}.dairportid = Flights_R3_dairports.dairportid
WHERE name2 = 't' and name4 = 't' and name1 = 3;'''

flights_sklearn_rf_1 = '''SELECT /*+PARALLEL({0})*/ Flights_S_routes{1}.airlineid, PREDICT flights_sklearn_rf_{2}( 
slatitude, slongitude, dlatitude, dlongitude, name1, name2, name4, acountry, active, 
scity, scountry, stimezone, sdst, dcity, dcountry, dtimezone, ddst) AS codeshare FROM 
Flights_S_routes{1} JOIN Flights_R1_airlines ON Flights_S_routes{1}.airlineid = Flights_R1_airlines.airlineid 
JOIN Flights_R2_sairports ON Flights_S_routes{1}.sairportid = Flights_R2_sairports.sairportid 
JOIN Flights_R3_dairports ON Flights_S_routes{1}.dairportid = Flights_R3_dairports.dairportid
WHERE name4 = 'f' and sdst = 'U' and ddst = 'E' and name1 > 1;'''

flights_sklearn_rf_10 = '''SELECT /*+PARALLEL({0})*/ Flights_S_routes{1}.airlineid, PREDICT flights_sklearn_rf_{2}( 
slatitude, slongitude, dlatitude, dlongitude, name1, name2, name4, acountry, active, 
scity, scountry, stimezone, sdst, dcity, dcountry, dtimezone, ddst) AS codeshare FROM 
Flights_S_routes{1} JOIN Flights_R1_airlines ON Flights_S_routes{1}.airlineid = Flights_R1_airlines.airlineid 
JOIN Flights_R2_sairports ON Flights_S_routes{1}.sairportid = Flights_R2_sairports.sairportid 
JOIN Flights_R3_dairports ON Flights_S_routes{1}.dairportid = Flights_R3_dairports.dairportid
WHERE name2 = 't' and name4 = 't' and name1 = 3;'''

flights_sklearn_rf_30 = '''SELECT /*+PARALLEL({0})*/ Flights_S_routes{1}.airlineid, PREDICT flights_sklearn_rf_{2}( 
slatitude, slongitude, dlatitude, dlongitude, name1, name2, name4, acountry, active, 
scity, scountry, stimezone, sdst, dcity, dcountry, dtimezone, ddst) AS codeshare FROM 
Flights_S_routes{1} JOIN Flights_R1_airlines ON Flights_S_routes{1}.airlineid = Flights_R1_airlines.airlineid 
JOIN Flights_R2_sairports ON Flights_S_routes{1}.sairportid = Flights_R2_sairports.sairportid 
JOIN Flights_R3_dairports ON Flights_S_routes{1}.dairportid = Flights_R3_dairports.dairportid
WHERE name2 = 't' and (sdst = 'A' or sdst = 'N');'''

flights_sklearn_rf_60 = '''SELECT /*+PARALLEL({0})*/ Flights_S_routes{1}.airlineid, PREDICT flights_sklearn_rf_{2}( 
slatitude, slongitude, dlatitude, dlongitude, name1, name2, name4, acountry, active, 
scity, scountry, stimezone, sdst, dcity, dcountry, dtimezone, ddst) AS codeshare FROM 
Flights_S_routes{1} JOIN Flights_R1_airlines ON Flights_S_routes{1}.airlineid = Flights_R1_airlines.airlineid 
JOIN Flights_R2_sairports ON Flights_S_routes{1}.sairportid = Flights_R2_sairports.sairportid 
JOIN Flights_R3_dairports ON Flights_S_routes{1}.dairportid = Flights_R3_dairports.dairportid
WHERE name4 = 'f' and name1 < 4;'''

flights_sklearn_rf_100 = '''SELECT /*+PARALLEL({0})*/ Flights_S_routes{1}.airlineid, PREDICT flights_sklearn_rf_{2}( 
slatitude, slongitude, dlatitude, dlongitude, name1, name2, name4, acountry, active, 
scity, scountry, stimezone, sdst, dcity, dcountry, dtimezone, ddst) AS codeshare FROM 
Flights_S_routes{1} JOIN Flights_R1_airlines ON Flights_S_routes{1}.airlineid = Flights_R1_airlines.airlineid 
JOIN Flights_R2_sairports ON Flights_S_routes{1}.sairportid = Flights_R2_sairports.sairportid 
JOIN Flights_R3_dairports ON Flights_S_routes{1}.dairportid = Flights_R3_dairports.dairportid;'''

flights_sklearn_rf_trees = '''SELECT /*+PARALLEL({0})*/ Flights_S_routes{1}.airlineid, PREDICT flights_sklearn_rf_{2}_trees{3}( 
slatitude, slongitude, dlatitude, dlongitude, name1, name2, name4, acountry, active, 
scity, scountry, stimezone, sdst, dcity, dcountry, dtimezone, ddst) AS codeshare FROM 
Flights_S_routes{1} JOIN Flights_R1_airlines ON Flights_S_routes{1}.airlineid = Flights_R1_airlines.airlineid 
JOIN Flights_R2_sairports ON Flights_S_routes{1}.sairportid = Flights_R2_sairports.sairportid 
JOIN Flights_R3_dairports ON Flights_S_routes{1}.dairportid = Flights_R3_dairports.dairportid
WHERE name2 = 't' and name4 = 't' and name1 = 3 limit 10000;'''

hospital_sklearn_lr = '''SELECT /*+PARALLEL({0})*/ eid, PREDICT hospital_sklearn_lr_{2}(
hematocrit, neutrophils, sodium, glucose, bloodureanitro, creatinine, bmi, pulse, respiration, 
secondarydiagnosisnonicd9, rcount, gender, dialysisrenalendstage, asthma, irondef, pneum, 
substancedependence, psychologicaldisordermajor, depress, psychother, fibrosisandother, malnutrition, hemo) 
AS lengthofstay FROM LengthOfStay{1} WHERE hematocrit > 10 AND neutrophils > 10 AND bloodureanitro < 16 AND pulse < 72;'''

hospital_tensorflow_mlp = '''SELECT /*+PARALLEL({0})*/ eid, PREDICT hospital_tensorflow_mlp_{2}(
hematocrit, neutrophils, sodium, glucose, bloodureanitro, creatinine, bmi, pulse, respiration, 
secondarydiagnosisnonicd9, rcount, gender, dialysisrenalendstage, asthma, irondef, pneum, 
substancedependence, psychologicaldisordermajor, depress, psychother, fibrosisandother, malnutrition, hemo) 
AS lengthofstay FROM LengthOfStay{1} WHERE hematocrit > 10 AND neutrophils > 10 AND bloodureanitro < 16 AND pulse < 72;'''

hospital_pytorch_mlp  = '''SELECT /*+PARALLEL({0})*/ eid, PREDICT hospital_pytorch_mlp_{2}(
hematocrit, neutrophils, sodium, glucose, bloodureanitro, creatinine, bmi, pulse, respiration, 
secondarydiagnosisnonicd9, rcount, gender, dialysisrenalendstage, asthma, irondef, pneum, 
substancedependence, psychologicaldisordermajor, depress, psychother, fibrosisandother, malnutrition, hemo) 
AS lengthofstay FROM LengthOfStay{1} WHERE hematocrit > 10 AND neutrophils > 10 AND bloodureanitro < 16 AND pulse < 72;'''

creditcard_lightgbm_gb = '''SELECT /*+PARALLEL({0})*/ Time, Amount, PREDICT creditcard_lightgbm_gb_{2}(
V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11, V12, V13, V14, V15, V16, V17, V18, V19, V20,
V21, V22, V23, V24, V25, V26, V27, V28, Amount) AS Class FROM Credit_Card{1} WHERE V1 > 1 AND V2 < 0.27 AND V3 > 0.3;'''

creditcard_lightgbm_gb_1 = '''SELECT /*+PARALLEL({0})*/ Time, Amount, PREDICT creditcard_lightgbm_gb_{2}(
V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11, V12, V13, V14, V15, V16, V17, V18, V19, V20,
V21, V22, V23, V24, V25, V26, V27, V28, Amount) AS Class FROM Credit_Card{1} WHERE V1 > 1.15 AND V2 < 0.2 AND V3 > 1;'''

creditcard_lightgbm_gb_10 = '''SELECT /*+PARALLEL({0})*/ Time, Amount, PREDICT creditcard_lightgbm_gb_{2}(
V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11, V12, V13, V14, V15, V16, V17, V18, V19, V20,
V21, V22, V23, V24, V25, V26, V27, V28, Amount) AS Class FROM Credit_Card{1} WHERE V1 > 1 AND V2 < 0.27 AND V3 > 0.3;'''

creditcard_lightgbm_gb_30 = '''SELECT /*+PARALLEL({0})*/ Time, Amount, PREDICT creditcard_lightgbm_gb_{2}(
V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11, V12, V13, V14, V15, V16, V17, V18, V19, V20,
V21, V22, V23, V24, V25, V26, V27, V28, Amount) AS Class FROM Credit_Card{1} WHERE V1 > 1.15 AND V2 < 0.23;'''

creditcard_lightgbm_gb_60 = '''SELECT /*+PARALLEL({0})*/ Time, Amount, PREDICT creditcard_lightgbm_gb_{2}(
V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11, V12, V13, V14, V15, V16, V17, V18, V19, V20,
V21, V22, V23, V24, V25, V26, V27, V28, Amount) AS Class FROM Credit_Card{1} WHERE V1 < 1.05;'''

creditcard_lightgbm_gb_100 = '''SELECT /*+PARALLEL({0})*/ Time, Amount, PREDICT creditcard_lightgbm_gb_{2}(
V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11, V12, V13, V14, V15, V16, V17, V18, V19, V20,
V21, V22, V23, V24, V25, V26, V27, V28, Amount) AS Class FROM Credit_Card{1};'''

creditcard_lightgbm_gb_trees = '''SELECT /*+PARALLEL({0})*/ Time, Amount, PREDICT creditcard_lightgbm_gb_{2}_trees{3}(
V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11, V12, V13, V14, V15, V16, V17, V18, V19, V20,
V21, V22, V23, V24, V25, V26, V27, V28, Amount) AS Class FROM Credit_Card{1} WHERE V1 > 1 AND V2 < 0.27 AND V3 > 0.3 limit 10000;'''

creditcard_xgboost_gb = '''SELECT /*+PARALLEL({0})*/ Time, Amount, PREDICT creditcard_xgboost_gb_{2}(
V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11, V12, V13, V14, V15, V16, V17, V18, V19, V20,
V21, V22, V23, V24, V25, V26, V27, V28, Amount) AS Class FROM Credit_Card{1} WHERE V1 > 1 AND V2 < 0.27 AND V3 > 0.3'''

creditcard_xgboost_gb_trees = '''SELECT /*+PARALLEL({0})*/ Time, Amount, PREDICT creditcard_xgboost_gb_{2}_trees{3}(
V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11, V12, V13, V14, V15, V16, V17, V18, V19, V20,
V21, V22, V23, V24, V25, V26, V27, V28, Amount) AS Class FROM Credit_Card{1} WHERE V1 > 1 AND V2 < 0.27 AND V3 > 0.3 limit 10000;'''

uc01 = '''select /*+PARALLEL({1})*/ ratio_tbl.o_customer_sk, PREDICT tpcx_ai_uc01_{0}_{2}(CAST(COALESCE(return_ratio,0) AS DOUBLE), CAST(COALESCE(frequency,0) AS DOUBLE)) as cluster
from (select o_customer_sk, avg(ratio) return_ratio 
from (select o_customer_sk, return_row_price_sum/row_price_sum ratio
from (select o_order_id, o_customer_sk, SUM(row_price) row_price_sum, SUM(return_row_price) return_row_price_sum, SUM(invoice_year) invoice_year_min 
from (select o_order_id, o_customer_sk, extract(year FROM cast(date0 as DATE)) invoice_year, quantity*price row_price, or_return_quantity*price return_row_price 
from (select o_order_id, o_customer_sk, date date0, li_product_id, price, quantity, or_return_quantity  
from (select * from Lineitem_{0} left join Order_Returns_{0} 
on Lineitem_{0}.li_order_id = Order_Returns_{0}.or_order_id 
and Lineitem_{0}.li_product_id = Order_Returns_{0}.or_product_id
) returns_data Join Order_o_{0} on returns_data.li_order_id=Order_o_{0}.o_order_id)) 
group by o_order_id, o_customer_sk)
)
group by o_customer_sk
) ratio_tbl 
join (select o_customer_sk, avg(o_order_id) frequency
from (select o_customer_sk, invoice_year_min, count(DISTINCT o_order_id) o_order_id
from (select o_order_id, o_customer_sk, SUM(row_price) row_price_sum, SUM(return_row_price) return_row_price_sum, SUM(invoice_year) invoice_year_min 
from (select o_order_id, o_customer_sk, extract(year FROM cast(date0 as DATE)) invoice_year, quantity*price row_price, or_return_quantity*price return_row_price 
from (select o_order_id, o_customer_sk, date date0, li_product_id, price, quantity, or_return_quantity  
from (select * from Lineitem_{0} left join Order_Returns_{0} 
on Lineitem_{0}.li_order_id = Order_Returns_{0}.or_order_id 
and Lineitem_{0}.li_product_id = Order_Returns_{0}.or_product_id
) returns_data Join Order_o_{0} on returns_data.li_order_id=Order_o_{0}.o_order_id)) 
group by o_order_id, o_customer_sk
)
group by o_customer_sk, invoice_year_min
) group by o_customer_sk
) frequency_tbl on ratio_tbl.o_customer_sk = frequency_tbl.o_customer_sk;'''

uc03 = '''select /*+PARALLEL({1})*/ store, department, PREDICT tpcx_ai_uc03_{0}_{2}(store, department) 
from (select store, department
from Order_o_{0} Join Lineitem_{0} on Order_o_{0}.o_order_id = Lineitem_{0}.li_order_id
Join Product_{0} on Lineitem_{0}.li_product_id = Product_{0}.p_product_id 
group by store, department);'''

uc04 = '''select /*+PARALLEL({1})*/ ID, PREDICT tpcx_ai_uc04_{0}_{2}(text) from Review_{0};'''

uc06 = '''select /*+PARALLEL({1})*/ serial_number, PREDICT tpcx_ai_uc06_{0}_{2}(smart_5_raw, smart_10_raw, smart_184_raw, 
smart_187_raw, smart_188_raw, smart_197_raw, smart_198_raw) as failure from Failures_{0}'''

uc07 = '''select /*+PARALLEL({1})*/ userID, productID, r, score 
from (select userID, productID, score, rank() OVER (PARTITION BY userID ORDER BY score) as r 
from (select userID, productID, PREDICT tpcx_ai_uc07_{0}_{2}(userID, productID) score 
from (select userID, productID 
from Product_Rating_{0}
group by userID, productID)))
where r <= 10;'''

uc10 = '''select /*+NO_REWRITE*/ /*+PARALLEL({1})*/ transactionID, PREDICT tpcx_ai_uc10_{0}_{2}(CAST(business_hour_norm AS DOUBLE), 
CAST(amount_norm AS DOUBLE)) AS isFraud from (select transactionID, amount/transaction_limit amount_norm, 
HOUR(STR_TO_DATE(time, '%Y-%m-%dT%H:%i'))/23 AS business_hour_norm from Financial_Account_{0} join 
Financial_Transactions_{0} on Financial_Account_{0}.fa_customer_sk = Financial_Transactions_{0}.senderID)'''

uc10_args = '''select /*+PARALLEL({1})*/ transactionID, CAST(business_hour_norm AS DOUBLE), 
CAST(amount_norm AS DOUBLE) from (select transactionID, amount/transaction_limit amount_norm, 
HOUR(STR_TO_DATE(time, '%Y-%m-%dT%H:%i'))/23 AS business_hour_norm from Financial_Account_{0} join 
Financial_Transactions_{0} on Financial_Account_{0}.fa_customer_sk = Financial_Transactions_{0}.senderID);'''

uc10_args2 = '''select transactionID, amount/transaction_limit amount_norm, 
time, STR_TO_DATE(time, '%Y-%m-%dT%H:%i') AS business_hour_norm from Financial_Account_{0} join 
Financial_Transactions_{0} on Financial_Account_{0}.fa_customer_sk = Financial_Transactions_{0}.senderID limit 100;'''

test_efficiency = '''select PREDICT test_efficiency(c1) from t1;'''
test_pytorch = '''select PREDICT test_torch(c1) from t1;'''
test_tensorflow = '''select PREDICT test_tensorflow(c1) from t1;'''

def set_query_raven(test_sql):
	parallel_num = 8
	scale_factor = "_1G"
	isInit = "uninit"
	for elem in [10, 20 , 30, 40 , 50]:
		scale_factor = f"_{elem}G"
		test_sql.append(expedia_sklearn_dt.format(parallel_num, scale_factor, isInit)) # Q2 ok
		test_sql.append(hospital_pytorch_mlp.format(parallel_num, scale_factor, isInit)) # Q4 ok
		test_sql.append(creditcard_lightgbm_gb.format(parallel_num, scale_factor, isInit)) # Q5 ok
	#test_sql.append(expedia_onnx_dt.format(parallel_num, scale_factor, isInit)) # Q1 ok
	# test_sql.append(expedia_sklearn_dt.format(parallel_num, scale_factor, isInit)) # Q2 ok
	# test_sql.append(flights_sklearn_rf.format(parallel_num, scale_factor, isInit)) # Q3 ok
	# test_sql.append(hospital_pytorch_mlp.format(parallel_num, scale_factor, isInit)) # Q4 ok
	# test_sql.append(creditcard_lightgbm_gb.format(parallel_num, scale_factor, isInit)) # Q5 ok
	# test_sql.append(creditcard_xgboost_gb.format(parallel_num, scale_factor, isInit)) # Q6 ok
	# test_sql.append(creditcard_xgboost_gb.format(parallel_num, scale_factor, "init")) # Q6 ok

	# test_sql.append(expedia_sklearn_dt_1.format(parallel_num, scale_factor, isInit)) # Q2 ok
	# test_sql.append(expedia_sklearn_dt_10.format(parallel_num, scale_factor, isInit)) # Q2 ok
	# test_sql.append(expedia_sklearn_dt_30.format(parallel_num, scale_factor, isInit)) # Q2 ok
	# test_sql.append(expedia_sklearn_dt_60.format(parallel_num, scale_factor, isInit)) # Q2 ok
	# test_sql.append(expedia_sklearn_dt_100.format(parallel_num, scale_factor, isInit)) # Q2 ok
# 
	# test_sql.append(flights_sklearn_rf_1.format(parallel_num, scale_factor, isInit)) # Q3 ok
	# test_sql.append(flights_sklearn_rf_10.format(parallel_num, scale_factor, isInit)) # Q3 ok
	# test_sql.append(flights_sklearn_rf_30.format(parallel_num, scale_factor, isInit)) # Q3 ok
	# test_sql.append(flights_sklearn_rf_60.format(parallel_num, scale_factor, isInit)) # Q3 ok
	# test_sql.append(flights_sklearn_rf_100.format(parallel_num, scale_factor, isInit)) # Q3 ok
# 
	# test_sql.append(creditcard_lightgbm_gb_1.format(parallel_num, scale_factor, isInit)) # Q5 ok
	# test_sql.append(creditcard_lightgbm_gb_10.format(parallel_num, scale_factor, isInit)) # Q5 ok
	# test_sql.append(creditcard_lightgbm_gb_30.format(parallel_num, scale_factor, isInit)) # Q5 ok
	# test_sql.append(creditcard_lightgbm_gb_60.format(parallel_num, scale_factor, isInit)) # Q5 ok
	# test_sql.append(creditcard_lightgbm_gb_100.format(parallel_num, scale_factor, isInit)) # Q5 ok


def set_query_raven_trees(test_sql):
	parallel_num = 1
	scale_factor = "_1G"
	isInit = "uninit"
	for elem in [10,50,100,500,1000]:
		test_sql.append(flights_sklearn_rf_trees.format(parallel_num, scale_factor, isInit, elem)) # Q3 ok
		test_sql.append(creditcard_lightgbm_gb_trees.format(parallel_num, scale_factor, isInit, elem)) # Q5 ok
		test_sql.append(creditcard_xgboost_gb_trees.format(parallel_num, scale_factor, isInit, elem)) # Q6 ok
	
def set_query_tpcx_ai(test_sql):
	parallel_num = 8
	scale_factor = "sf10"
	isInit = "uninit"
	for elem in [100]:
		scale_factor = f"sf{elem}"
		# test_sql.append(uc03.format(scale_factor, parallel_num, isInit)) # Q8
		# test_sql.append(uc07.format(scale_factor, parallel_num, isInit)) # Q11
		test_sql.append(uc10.format(scale_factor, parallel_num, isInit)) # Q12
	
	# test_sql.append(uc01.format(scale_factor, parallel_num, isInit)) # Q7
	# test_sql.append(uc03.format(scale_factor, parallel_num, isInit)) # Q8
	# test_sql.append(uc04.format(scale_factor, parallel_num, isInit)) # Q9
	# test_sql.append(uc06.format(scale_factor, parallel_num, isInit)) # Q10
	# test_sql.append(uc07.format(scale_factor, parallel_num, isInit)) # Q11
	# test_sql.append(uc10.format(scale_factor, parallel_num, isInit)) # Q12

def set_query_test(test_sql):
	#test_sql.append(test_efficiency)
	#test_sql.append(test_pytorch)
	test_sql.append(test_tensorflow)

raven_on = False
tpcx_ai_on = True
test_on = False

for i in range(repeat):

	if (raven_on):
		# raven connection
		conn = pymysql.connect(host="127.0.0.1", port=2881, user="root", passwd="ulAFVBT0D4GDfMizqJXJ", db="raven")
		cur = conn.cursor()

		try:
			test_sql = []
			time_stat = 0
			cur.execute(trace_on) # open trace
			set_query_raven(test_sql)
			# set_query_raven_trees(test_sql)
			for i in test_sql:
				print(count)
				print(i)
				time_consuming = run_sql(cur, i)
				df = pd.DataFrame({'query': [i[:100]], 'execute': [time_consuming], 'time': [time.asctime()]})
				df.to_csv(path, index=True, mode='a', header=None)
				count += 1

		finally:
			cur.close()
			conn.close()
	
	if (tpcx_ai_on):
		# tpcx_ai connection
		conn = pymysql.connect(host="127.0.0.1", port=2881, user="root", passwd="ulAFVBT0D4GDfMizqJXJ", db="tpcx_ai")
		cur = conn.cursor()

		try:
			test_sql = []
			time_stat = 0
			cur.execute(trace_on) # open trace
			set_query_tpcx_ai(test_sql)
			for i in test_sql:
				print(i)
				print(count)
				time_consuming = run_sql(cur, i)
				df = pd.DataFrame({'query': [i[:100]], 'execute': [time_consuming], 'time': [time.asctime()]})
				df.to_csv(path, index=True, mode='a', header=None)
				count += 1

		finally:
			cur.close()
			conn.close()

	if (test_on):
		# test connection
		conn = pymysql.connect(host="127.0.0.1", port=2881, user="root", passwd="ulAFVBT0D4GDfMizqJXJ", db="test")
		cur = conn.cursor()

		try:
			test_sql = []
			time_stat = 0
			cur.execute(trace_on) # open trace
			set_query_test(test_sql)
			for i in test_sql:
				time_consuming = run_sql(cur, i)
				df = pd.DataFrame({'query': [i[:100]], 'execute': [time_consuming], 'time': [time.asctime()]})
				df.to_csv(path, index=True, mode='a', header=None)

		finally:
			cur.close()
			conn.close()
