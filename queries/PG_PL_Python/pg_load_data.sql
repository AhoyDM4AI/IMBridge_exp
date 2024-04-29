DROP TABLE IF EXISTS Order_Returns;
DROP TABLE IF EXISTS Order_o;
DROP TABLE IF EXISTS Lineitem;
DROP TABLE IF EXISTS Product;
DROP TABLE IF EXISTS Review;
DROP TABLE IF EXISTS Product_Rating;
DROP TABLE IF EXISTS Failures;
DROP TABLE IF EXISTS Financial_Account;
DROP TABLE IF EXISTS Financial_Transactions;

-- tpc
CREATE TABLE IF NOT EXISTS Order_Returns(or_order_id INTEGER, or_product_id INTEGER, or_return_quantity INTEGER);

CREATE TABLE IF NOT EXISTS Order_o(o_order_id INTEGER, o_customer_sk INTEGER, weekday VARCHAR(32), date VARCHAR(32), store INTEGER);

CREATE TABLE IF NOT EXISTS Lineitem(li_order_id INTEGER, li_product_id INTEGER, quantity INTEGER, price REAL);

CREATE TABLE IF NOT EXISTS Product(p_product_id INTEGER, name VARCHAR(32), department VARCHAR(32));

CREATE TABLE IF NOT EXISTS Review(ID INTEGER, text VARCHAR(65535));

CREATE TABLE IF NOT EXISTS Product_Rating (userID INTEGER, productID INTEGER);

CREATE TABLE IF NOT EXISTS Failures (date VARCHAR(32), serial_number INTEGER, model VARCHAR(32), smart_5_raw REAL, smart_10_raw REAL,
    smart_184_raw REAL, smart_187_raw REAL, smart_188_raw REAL, smart_197_raw REAL, smart_198_raw REAL);

CREATE TABLE IF NOT EXISTS Financial_Account (fa_customer_sk INTEGER, transaction_limit REAL);

CREATE TABLE IF NOT EXISTS Financial_Transactions (amount REAL, IBAN VARCHAR(32), senderID INTEGER, receiverID VARCHAR(32), transactionID VARCHAR(32), time VARCHAR(32));

\COPY Order_Returns FROM './tpcxai_datasets/serving/order_returns.csv' CSV HEADER;
\COPY Order_o FROM './tpcxai_datasets/serving/order.csv' CSV HEADER;
\COPY Lineitem FROM './tpcxai_datasets/serving/lineitem.csv' CSV HEADER;
\COPY Product FROM './tpcxai_datasets/serving/product.csv' CSV HEADER;
\COPY Review FROM './tpcxai_datasets/serving/Review.psv' DELIMITER '|' CSV HEADER;
\COPY Product_Rating FROM './tpcxai_datasets/serving/ProductRating.csv' CSV HEADER;
\COPY Failures FROM './tpcxai_datasets/serving/failures.csv' CSV HEADER;
\COPY Financial_Account FROM './tpcxai_datasets/serving/financial_account.csv' CSV HEADER;
\COPY Financial_Transactions FROM './tpcxai_datasets/serving/financial_transactions.csv' CSV HEADER;

-- pf
DROP TABLE IF EXISTS Expedia_S_listings_extension;
DROP TABLE IF EXISTS Expedia_R1_hotels;
DROP TABLE IF EXISTS Expedia_R2_searches;
DROP TABLE IF EXISTS Flights_S_routes_extension;
DROP TABLE IF EXISTS Flights_R1_airlines;
DROP TABLE IF EXISTS Flights_R2_sairports;
DROP TABLE IF EXISTS Flights_R3_dairports;
DROP TABLE IF EXISTS LengthOfStay_extension;
DROP TABLE IF EXISTS LengthOfStay;
DROP TABLE IF EXISTS Credit_Card_extension;
DROP TABLE IF EXISTS Credit_Card;

CREATE TABLE IF NOT EXISTS Expedia_S_listings_extension(srch_id VARCHAR(32), prop_id VARCHAR(32), position VARCHAR(32), prop_location_score1 REAL,
    prop_location_score2 REAL, prop_log_historical_price REAL, price_usd REAL, promotion_flag BOOLEAN, orig_destination_distance REAL);

CREATE TABLE IF NOT EXISTS Expedia_R1_hotels(prop_id VARCHAR(32), prop_country_id VARCHAR(32), prop_starrating INTEGER, prop_review_score REAL, prop_brand_bool BOOLEAN,
    count_clicks INTEGER, avg_bookings_usd REAL, stdev_bookings_usd REAL, count_bookings INTEGER);

CREATE TABLE IF NOT EXISTS Expedia_R2_searches(srch_id VARCHAR(32), year VARCHAR(32), month VARCHAR(32), weekofyear VARCHAR(32), time VARCHAR(32), site_id VARCHAR(32),
    visitor_location_country_id VARCHAR(32), srch_destination_id VARCHAR(32), srch_length_of_stay INTEGER, srch_booking_window INTEGER,
    srch_adults_count INTEGER, srch_children_count INTEGER, srch_room_count INTEGER, srch_saturday_night_bool BOOLEAN, random_bool BOOLEAN);

CREATE TABLE IF NOT EXISTS Flights_S_routes_extension(idx INTEGER, airlineid VARCHAR(32), sairportid VARCHAR(32), dairportid VARCHAR(32), codeshare VARCHAR(32));

CREATE TABLE IF NOT EXISTS Flights_R1_airlines(airlineid VARCHAR(32), name1 INTEGER, name2 VARCHAR(32), name4 VARCHAR(32), acountry VARCHAR(64), active VARCHAR(32));

CREATE TABLE IF NOT EXISTS Flights_R2_sairports(sairportid VARCHAR(32), scity VARCHAR(32), scountry VARCHAR(32),
    slatitude REAL, slongitude REAL, stimezone INTEGER, sdst VARCHAR(32));

CREATE TABLE IF NOT EXISTS Flights_R3_dairports(dairportid VARCHAR(32), dcity VARCHAR(32),
    dcountry VARCHAR(32), dlatitude REAL, dlongitude REAL, dtimezone INTEGER, ddst VARCHAR(32));

CREATE TABLE IF NOT EXISTS LengthOfStay_extension(eid INTEGER, vdate VARCHAR(32), rcount VARCHAR(32), gender VARCHAR(32), dialysisrenalendstage BOOLEAN,
    asthma BOOLEAN, irondef BOOLEAN, pneum BOOLEAN, substancedependence BOOLEAN, psychologicaldisordermajor BOOLEAN, depress BOOLEAN,
    psychother BOOLEAN, fibrosisandother BOOLEAN, malnutrition BOOLEAN, hemo BOOLEAN, hematocrit REAL, neutrophils REAL, sodium REAL,
    glucose REAL, bloodureanitro REAL, creatinine REAL, bmi REAL,
    pulse INTEGER, respiration REAL, secondarydiagnosisnonicd9 INTEGER,
    discharged VARCHAR(32), facid VARCHAR(32), lengthofstay INTEGER);

CREATE TABLE IF NOT EXISTS LengthOfStay(eid INTEGER, vdate VARCHAR(32), rcount VARCHAR(32), gender VARCHAR(32), dialysisrenalendstage BOOLEAN,
    asthma BOOLEAN, irondef BOOLEAN, pneum BOOLEAN, substancedependence BOOLEAN, psychologicaldisordermajor BOOLEAN, depress BOOLEAN,
    psychother BOOLEAN, fibrosisandother BOOLEAN, malnutrition BOOLEAN, hemo BOOLEAN, hematocrit REAL, neutrophils REAL, sodium REAL,
    glucose REAL, bloodureanitro REAL, creatinine REAL, bmi REAL,
    pulse INTEGER, respiration REAL, secondarydiagnosisnonicd9 INTEGER,
    discharged VARCHAR(32), facid VARCHAR(32), lengthofstay INTEGER);

CREATE TABLE IF NOT EXISTS Credit_Card_extension(Time VARCHAR(256), V1 REAL, V2 REAL, V3 REAL, V4 REAL, V5 REAL, V6 REAL, V7 REAL, V8 REAL, V9 REAL,
                                                 V10 REAL, V11 REAL, V12 REAL, V13 REAL, V14 REAL, V15 REAL, V16 REAL, V17 REAL, V18 REAL, V19 REAL,
                                                 V20 REAL, V21 REAL, V22 REAL, V23 REAL, V24 REAL, V25 REAL, V26 REAL, V27 REAL, V28 REAL, Amount REAL,
                                                 Class BOOLEAN);

CREATE TABLE IF NOT EXISTS Credit_Card(Time VARCHAR(256), V1 REAL, V2 REAL, V3 REAL, V4 REAL, V5 REAL, V6 REAL, V7 REAL, V8 REAL, V9 REAL,
                                                 V10 REAL, V11 REAL, V12 REAL, V13 REAL, V14 REAL, V15 REAL, V16 REAL, V17 REAL, V18 REAL, V19 REAL,
                                                 V20 REAL, V21 REAL, V22 REAL, V23 REAL, V24 REAL, V25 REAL, V26 REAL, V27 REAL, V28 REAL, Amount REAL,
                                                 Class BOOLEAN);

\COPY Expedia_S_listings_extension FROM './public_datasets/Expedia/S_listings_extension.csv' CSV HEADER;
\COPY Expedia_R1_hotels FROM './public_datasets/Expedia/R1_hotels_2.csv' CSV HEADER;
\COPY Expedia_R2_searches FROM './public_datasets/Expedia/R2_searches.csv' CSV HEADER;
\COPY Flights_S_routes_extension FROM './public_datasets/Flights/S_routes_1G.csv' CSV HEADER;
\COPY Flights_R1_airlines FROM './public_datasets/Flights/R1_airlines.csv' CSV HEADER;
\COPY Flights_R2_sairports FROM './public_datasets/Flights/R2_sairports.csv' CSV HEADER;
\COPY Flights_R3_dairports FROM './public_datasets/Flights/R3_dairports.csv' CSV HEADER;
\COPY LengthOfStay_extension FROM './public_datasets/Hospital/hospital_1G.csv' CSV HEADER;
\COPY LengthOfStay FROM './public_datasets/Hospital/LengthOfStay.csv' CSV HEADER;
\COPY Credit_Card_extension FROM './public_datasets/Credit_Card/creditcard_extension.csv' CSV HEADER;
\COPY Credit_Card FROM './public_datasets/Credit_Card/creditcard.csv' CSV HEADER;
