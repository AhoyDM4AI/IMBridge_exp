SET GLOBAL SECURE_FILE_PRIV = '';

GRANT FILE ON *.* TO root;

set global ob_trx_idle_timeout=36000000000;
set global ob_trx_timeout=36000000000;
set global ob_query_timeout=36000000000;

CREATE DATABASE raven;

USE raven;

CREATE TABLE Expedia_S_listings_1G(srch_id VARCHAR(32), prop_id VARCHAR(32), position VARCHAR(32), prop_location_score1 DOUBLE, 
prop_location_score2 DOUBLE, prop_log_historical_price DOUBLE, price_usd DOUBLE, promotion_flag BOOLEAN, orig_destination_distance DOUBLE);

CREATE TABLE Expedia_R1_hotels_1G(prop_id VARCHAR(32), prop_country_id VARCHAR(32), prop_starrating INTEGER, prop_review_score DOUBLE, 
prop_brand_bool BOOLEAN, count_clicks INTEGER, avg_bookings_usd DOUBLE, stdev_bookings_usd DOUBLE, count_bookings INTEGER);

CREATE TABLE Expedia_R2_searches_1G(srch_id VARCHAR(32), year VARCHAR(32), month VARCHAR(32), weekofyear VARCHAR(32), time VARCHAR(32), 
site_id VARCHAR(32), visitor_location_country_id VARCHAR(32), srch_destination_id VARCHAR(32), srch_length_of_stay INTEGER, 
srch_booking_window INTEGER, srch_adults_count INTEGER, srch_children_count INTEGER, srch_room_count INTEGER, srch_saturday_night_bool BOOLEAN, random_bool BOOLEAN);

LOAD DATA
/*+ PARALLEL(16) */
INFILE '{HOME_PATH}/workload/raven_datasets/Expedia/S_listings_1G.csv'
INTO TABLE Expedia_S_listings_10G
FIELDS TERMINATED BY ',' ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;

LOAD DATA
/*+ PARALLEL(16) */
INFILE '{HOME_PATH}/workload/raven_datasets/Expedia/R1_hotels_1G.csv'
INTO TABLE Expedia_R1_hotels
FIELDS TERMINATED BY ',' ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;

LOAD DATA
/*+ PARALLEL(16) */
INFILE '{HOME_PATH}/workload/raven_datasets/Expedia/R2_searches_1G.csv'
INTO TABLE Expedia_R2_searches
FIELDS TERMINATED BY ',' ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;

CREATE TABLE Flights_S_routes_1G(airlineid VARCHAR(32), sairportid VARCHAR(32), dairportid VARCHAR(32));

CREATE TABLE Flights_R1_airlines_1G(airlineid VARCHAR(32), name1 INTEGER, name2 VARCHAR(32), name4 VARCHAR(32), acountry VARCHAR(64), active VARCHAR(32));

CREATE TABLE Flights_R2_sairports_1G(sairportid VARCHAR(32), scity VARCHAR(32), scountry VARCHAR(32), slatitude DOUBLE, slongitude DOUBLE, stimezone INTEGER, sdst VARCHAR(32));

CREATE TABLE Flights_R3_dairports_1G(dairportid VARCHAR(32), dcity VARCHAR(32), dcountry VARCHAR(32), dlatitude DOUBLE, dlongitude DOUBLE, dtimezone INTEGER, ddst VARCHAR(32));

LOAD DATA
/*+ PARALLEL(16) */
INFILE '{HOME_PATH}/workload/raven_datasets/Flights/S_routes_1G.csv'
INTO TABLE Flights_S_routes_1G
FIELDS TERMINATED BY ',' ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS
(airlineid, sairportid, dairportid);

LOAD DATA
/*+ PARALLEL(16) */
INFILE '{HOME_PATH}/workload/raven_datasets/Flights/R1_airlines_1G.csv'
INTO TABLE Flights_R1_airlines_1G
FIELDS TERMINATED BY ',' ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;

LOAD DATA
/*+ PARALLEL(16) */
INFILE '{HOME_PATH}/workload/raven_datasets/Flights/R2_sairports_1G.csv'
INTO TABLE Flights_R2_sairports_1G
FIELDS TERMINATED BY ',' ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;

LOAD DATA
/*+ PARALLEL(16) */
INFILE '{HOME_PATH}/workload/raven_datasets/Flights/R3_dairports_1G.csv'
INTO TABLE Flights_R3_dairports_1G
FIELDS TERMINATED BY ',' ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;

CREATE TABLE LengthOfStay_1G(eid INTEGER, vdate VARCHAR(32), rcount VARCHAR(32), gender VARCHAR(32), dialysisrenalendstage BOOLEAN, 
asthma BOOLEAN, irondef BOOLEAN, pneum BOOLEAN, substancedependence BOOLEAN, psychologicaldisordermajor BOOLEAN, depress BOOLEAN, 
psychother BOOLEAN, fibrosisandother BOOLEAN, malnutrition BOOLEAN, hemo BOOLEAN, hematocrit DOUBLE, neutrophils DOUBLE, sodium DOUBLE, 
glucose DOUBLE, bloodureanitro DOUBLE, creatinine DOUBLE, bmi DOUBLE, pulse INTEGER, respiration DOUBLE, secondarydiagnosisnonicd9 INTEGER, 
discharged VARCHAR(32), facid VARCHAR(32), lengthofstay INTEGER);

LOAD DATA
/*+ PARALLEL(16) */
INFILE '{HOME_PATH}/workload/raven_datasets/Hospital/hospital_1G.csv'
INTO TABLE LengthOfStay_1G
FIELDS TERMINATED BY ',' ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;

CREATE TABLE Credit_Card_1G(Time INTEGER, V1 DOUBLE, V2 DOUBLE, V3 DOUBLE, V4 DOUBLE, V5 DOUBLE, V6 DOUBLE, V7 DOUBLE, V8 DOUBLE, V9 DOUBLE, 
V10 DOUBLE, V11 DOUBLE, V12 DOUBLE, V13 DOUBLE, V14 DOUBLE, V15 DOUBLE, V16 DOUBLE, V17 DOUBLE, V18 DOUBLE, V19 DOUBLE, V20 DOUBLE, V21 DOUBLE, 
V22 DOUBLE, V23 DOUBLE, V24 DOUBLE, V25 DOUBLE, V26 DOUBLE, V27 DOUBLE, V28 DOUBLE, Amount DOUBLE, Class BOOLEAN);

LOAD DATA
/*+ PARALLEL(16) */
INFILE '{HOME_PATH}/workload/raven_datasets/Credit_Card/creditcard_1G.csv'
INTO TABLE Credit_Card_1G
FIELDS TERMINATED BY ',' ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;
