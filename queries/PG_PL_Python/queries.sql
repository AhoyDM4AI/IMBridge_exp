SET statement_timeout = 10800000;
--q1
SELECT 1;
EXPLAIN ANALYZE SELECT Expedia_R1_hotels.prop_id, Expedia_R2_searches.srch_id, udf1(prop_location_score1, prop_location_score2, prop_log_historical_price, price_usd,
                           orig_destination_distance, prop_review_score, avg_bookings_usd, stdev_bookings_usd,
                           position, prop_country_id, prop_starrating, cast(prop_brand_bool as INTEGER), count_clicks, count_bookings,
                           year, month, weekofyear, time, site_id, visitor_location_country_id, srch_destination_id,
                           srch_length_of_stay, srch_booking_window, srch_adults_count, srch_children_count,
                           srch_room_count, cast(srch_saturday_night_bool as INTEGER), cast(random_bool as INTEGER))
FROM Expedia_S_listings_extension JOIN Expedia_R1_hotels ON Expedia_S_listings_extension.prop_id = Expedia_R1_hotels.prop_id
JOIN Expedia_R2_searches ON Expedia_S_listings_extension.srch_id = Expedia_R2_searches.srch_id
WHERE prop_location_score1 > 1 and prop_location_score2 > 0.1
and prop_log_historical_price > 4 and count_bookings > 5
and srch_booking_window > 10 and srch_length_of_stay > 1;

--q2
SELECT 2;
EXPLAIN ANALYZE SELECT Expedia_R1_hotels.prop_id, Expedia_R2_searches.srch_id, udf2(prop_location_score1, prop_location_score2, prop_log_historical_price, price_usd,
                           orig_destination_distance, prop_review_score, avg_bookings_usd, stdev_bookings_usd,
                           position, prop_country_id, prop_starrating, cast(prop_brand_bool as INTEGER), count_clicks, count_bookings,
                           year, month, weekofyear, time, site_id, visitor_location_country_id, srch_destination_id,
                           srch_length_of_stay, srch_booking_window, srch_adults_count, srch_children_count,
                           srch_room_count, cast(srch_saturday_night_bool as INTEGER), cast(random_bool as INTEGER))
FROM Expedia_S_listings_extension JOIN Expedia_R1_hotels ON Expedia_S_listings_extension.prop_id = Expedia_R1_hotels.prop_id
JOIN Expedia_R2_searches ON Expedia_S_listings_extension.srch_id = Expedia_R2_searches.srch_id
WHERE prop_location_score1 > 1 and prop_location_score2 > 0.1
and prop_log_historical_price > 4 and count_bookings > 5
and srch_booking_window > 10 and srch_length_of_stay > 1;

--q3
SELECT 3;
EXPLAIN ANALYZE SELECT Flights_S_routes_extension.airlineid, Flights_S_routes_extension.sairportid, Flights_S_routes_extension.dairportid,
 udf3(slatitude, slongitude, dlatitude, dlongitude, name1, name2, name4, acountry, active,
 scity, scountry, stimezone, sdst, dcity, dcountry, dtimezone, ddst) AS codeshare
 FROM Flights_S_routes_extension JOIN Flights_R1_airlines ON Flights_S_routes_extension.airlineid = Flights_R1_airlines.airlineid
 JOIN Flights_R2_sairports ON Flights_S_routes_extension.sairportid = Flights_R2_sairports.sairportid JOIN Flights_R3_dairports
 ON Flights_S_routes_extension.dairportid = Flights_R3_dairports.dairportid
WHERE name2 = 't' and name4 = 't' and name1 > 2.8;

--q4
SELECT 4;
EXPLAIN ANALYZE SELECT eid, udf4(hematocrit, neutrophils, sodium, glucose, bloodureanitro, creatinine, bmi, pulse,
 respiration, secondarydiagnosisnonicd9, rcount, gender, cast(dialysisrenalendstage as INTEGER), cast(asthma as INTEGER),
  cast(irondef as INTEGER), cast(pneum as INTEGER), cast(substancedependence as INTEGER),
   cast(psychologicaldisordermajor as INTEGER), cast(depress as INTEGER), cast(psychother as INTEGER),
    cast(fibrosisandother as INTEGER), cast(malnutrition as INTEGER), cast(hemo as INTEGER)) AS lengthofstay
   FROM LengthOfStay_extension WHERE hematocrit > 10 AND neutrophils > 10 AND bloodureanitro < 20 AND pulse < 70;

--q5
SELECT 5;
EXPLAIN ANALYZE SELECT Time, Amount, udf5(V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11, V12, V13, V14, V15,
 V16, V17, V18, V19, V20, V21, V22, V23, V24, V25, V26, V27, V28, Amount) AS Class FROM Credit_Card_extension
 WHERE V1 > 1 AND V2 < 0.27 AND V3 > 0.3;

--q6
SELECT 6;
EXPLAIN ANALYZE SELECT Time, Amount, udf6(V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11, V12, V13, V14, V15,
 V16, V17, V18, V19, V20, V21, V22, V23, V24, V25, V26, V27, V28, Amount) AS Class FROM Credit_Card_extension
 WHERE V1 > 1 AND V2 < 0.27 AND V3 > 0.3;
