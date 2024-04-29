-- Select Expedia Staged Prediction
SELECT Expedia_S_listings_1G.srch_id, Expedia_S_listings_1G.prop_id, PREDICT expedia_sklearn_dt_staged(prop_location_score1, prop_location_score2, 
prop_log_historical_price, price_usd, orig_destination_distance, prop_review_score, avg_bookings_usd, stdev_bookings_usd,  position, 
prop_country_id, prop_starrating, prop_brand_bool, count_clicks,  count_bookings, year, month, weekofyear, time, site_id, 
visitor_location_country_id, srch_destination_id, srch_length_of_stay, srch_booking_window, srch_adults_count,  srch_children_count, 
srch_room_count, srch_saturday_night_bool, random_bool) AS promotion_flag 
FROM Expedia_S_listings_1G JOIN Expedia_R1_hotels_1G ON Expedia_S_listings_1G.prop_id = Expedia_R1_hotels_1G.prop_id 
JOIN Expedia_R2_searches_1G ON Expedia_S_listings_1G.srch_id = Expedia_R2_searches_1G.srch_id;

/* 1.00%
WHERE prop_location_score1 > 1 and prop_location_score2 > 0.1 and prop_log_historical_price > 4 and count_bookings > 5 and srch_booking_window > 4 and srch_length_of_stay > 5;
/*

/* 10.00%
WHERE prop_location_score1 > 1 and prop_location_score2 > 0.1 and prop_log_historical_price > 4 and count_bookings > 5 and srch_booking_window > 4 and srch_length_of_stay > 1; # 10.00%
*/

/* 30.02%
WHERE prop_location_score1 > 1.5 and prop_location_score2 > 0.1; # 10.00%
*/

/* 60.59%
WHERE Expedia_S_listings_extension.srch_id = Expedia_R2_searches.srch_id WHERE count_bookings > 5 and srch_booking_window > 1; # 10.00%