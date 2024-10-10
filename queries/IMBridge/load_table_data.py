import pymysql
import json
import time
import pandas as pd
import os

trace_on = "SET ob_enable_show_trace = 1;"
show_trace = "SHOW TRACE;"
plan_flush = "ALTER SYSTEM FLUSH PLAN CACHE;"

prefix = "/home/test/experiments"

output_path = "./load_table_data.log"

def run_sql(cur, sql):
	cur.execute(plan_flush)
	cur.execute(sql)
	#time_consuming = analysis_trace(cur)
	#return time_consuming
	rows = cur.fetchall()
	for row in rows:
		print(row)

def analysis_trace(cur):
	cur.execute(show_trace)
	trace = cur.fetchone()
	if trace is not None:
		return trace[2]
	else:
		return -1

# raven
# expedia
Expedia_S_listings = "CREATE TABLE IF NOT EXISTS Expedia_S_listings{}(srch_id VARCHAR(32), prop_id VARCHAR(32), position VARCHAR(32), prop_location_score1 DOUBLE, prop_location_score2 DOUBLE, prop_log_historical_price DOUBLE, price_usd DOUBLE, promotion_flag BOOLEAN, orig_destination_distance DOUBLE);"
Expedia_R1_hotels = "CREATE TABLE IF NOT EXISTS Expedia_R1_hotels(prop_id VARCHAR(32), prop_country_id VARCHAR(32), prop_starrating INTEGER, prop_review_score DOUBLE, prop_brand_bool BOOLEAN, count_clicks INTEGER, avg_bookings_usd DOUBLE, stdev_bookings_usd DOUBLE, count_bookings INTEGER);"
Expedia_R2_searches = "CREATE TABLE IF NOT EXISTS Expedia_R2_searches(srch_id VARCHAR(32), year VARCHAR(32), month VARCHAR(32), weekofyear VARCHAR(32), time VARCHAR(32), site_id VARCHAR(32), visitor_location_country_id VARCHAR(32), srch_destination_id VARCHAR(32), srch_length_of_stay INTEGER, srch_booking_window INTEGER, srch_adults_count INTEGER, srch_children_count INTEGER, srch_room_count INTEGER, srch_saturday_night_bool BOOLEAN, random_bool BOOLEAN);"

# flights
Flights_S_routes = "CREATE TABLE IF NOT EXISTS Flights_S_routes{}(fid INTEGER, airlineid VARCHAR(32), sairportid VARCHAR(32), dairportid VARCHAR(32));"
Flights_R1_airlines = "CREATE TABLE IF NOT EXISTS Flights_R1_airlines(airlineid VARCHAR(32), name1 INTEGER, name2 VARCHAR(32), name4 VARCHAR(32), acountry VARCHAR(64), active VARCHAR(32));"
Flights_R2_sairports = "CREATE TABLE IF NOT EXISTS Flights_R2_sairports(sairportid VARCHAR(32), scity VARCHAR(32), scountry VARCHAR(32), slatitude DOUBLE, slongitude DOUBLE, stimezone INTEGER, sdst VARCHAR(32));"
Flights_R3_dairports = "CREATE TABLE IF NOT EXISTS Flights_R3_dairports(dairportid VARCHAR(32), dcity VARCHAR(32), dcountry VARCHAR(32), dlatitude DOUBLE, dlongitude DOUBLE, dtimezone INTEGER, ddst VARCHAR(32));"

# hospital
LengthOfStay = "CREATE TABLE IF NOT EXISTS LengthOfStay{}(eid INTEGER, vdate VARCHAR(32), rcount VARCHAR(32), gender VARCHAR(32), dialysisrenalendstage BOOLEAN, asthma BOOLEAN, irondef BOOLEAN, pneum BOOLEAN, substancedependence BOOLEAN, psychologicaldisordermajor BOOLEAN, depress BOOLEAN, psychother BOOLEAN, fibrosisandother BOOLEAN, malnutrition BOOLEAN, hemo BOOLEAN, hematocrit DOUBLE, neutrophils DOUBLE, sodium DOUBLE, glucose DOUBLE, bloodureanitro DOUBLE, creatinine DOUBLE, bmi DOUBLE, pulse INTEGER, respiration DOUBLE, secondarydiagnosisnonicd9 INTEGER, discharged VARCHAR(32), facid VARCHAR(32), lengthofstay INTEGER);"

# credit card
Credit_Card = "CREATE TABLE IF NOT EXISTS Credit_Card{}(Time INTEGER, V1 DOUBLE, V2 DOUBLE, V3 DOUBLE, V4 DOUBLE, V5 DOUBLE, V6 DOUBLE, V7 DOUBLE, V8 DOUBLE, V9 DOUBLE, V10 DOUBLE, V11 DOUBLE, V12 DOUBLE, V13 DOUBLE, V14 DOUBLE, V15 DOUBLE, V16 DOUBLE, V17 DOUBLE, V18 DOUBLE, V19 DOUBLE, V20 DOUBLE, V21 DOUBLE, V22 DOUBLE, V23 DOUBLE, V24 DOUBLE, V25 DOUBLE, V26 DOUBLE, V27 DOUBLE, V28 DOUBLE, Amount DOUBLE, Class BOOLEAN);"

# INFILE table data
# load expedia
load_Expedia_S_listings = '''LOAD DATA
/*+ PARALLEL(64) */
INFILE '{0}/test_raven/Expedia/S_listings{1}.csv'
INTO TABLE Expedia_S_listings{1}
FIELDS TERMINATED BY ',' ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;'''

load_Expedia_R1_hotels = '''LOAD DATA
/*+ PARALLEL(64) */
INFILE '{}/test_raven/Expedia/R1_hotels_2.csv'
INTO TABLE Expedia_R1_hotels
FIELDS TERMINATED BY ',' ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;'''

load_Expedia_R2_searches = '''LOAD DATA
/*+ PARALLEL(64) */
INFILE '{}/test_raven/Expedia/R2_searches.csv'
INTO TABLE Expedia_R2_searches
FIELDS TERMINATED BY ',' ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;'''

# load flights
load_Flights_S_routes = '''LOAD DATA
/*+ PARALLEL(64) */
INFILE '{0}/test_raven/Flights/S_routes{1}.csv'
INTO TABLE Flights_S_routes{1}
FIELDS TERMINATED BY ',' ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;'''

load_Flights_R1_airlines = '''LOAD DATA
/*+ PARALLEL(64) */
INFILE '{}/test_raven/Flights/R1_airlines.csv'
INTO TABLE Flights_R1_airlines
FIELDS TERMINATED BY ',' ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;'''

load_Flights_R2_sairports = '''LOAD DATA
/*+ PARALLEL(64) */
INFILE '{}/test_raven/Flights/R2_sairports.csv'
INTO TABLE Flights_R2_sairports
FIELDS TERMINATED BY ',' ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;'''

load_Flights_R3_dairports = '''LOAD DATA
/*+ PARALLEL(64) */
INFILE '{}/test_raven/Flights/R3_dairports.csv'
INTO TABLE Flights_R3_dairports
FIELDS TERMINATED BY ',' ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;'''

# load hospital
load_LengthOfStay = '''LOAD DATA
/*+ PARALLEL(64) */
INFILE '{0}/test_raven/Hospital/LengthOfStay{1}.csv'
INTO TABLE LengthOfStay{1}
FIELDS TERMINATED BY ',' ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;'''

load_Credit_Card = '''LOAD DATA
/*+ PARALLEL(64) */
INFILE '{0}/test_raven/Credit_Card/creditcard{1}.csv'
INTO TABLE Credit_Card{1}
FIELDS TERMINATED BY ',' ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;'''

# load credit card

# tpcx_ai 
# uc01
Order_Returns = "CREATE TABLE IF NOT EXISTS Order_Returns_{}(or_order_id INTEGER, or_product_id INTEGER, or_return_quantity INTEGER);"

# uc03
Order_o = "CREATE TABLE IF NOT EXISTS Order_o_{}(o_order_id INTEGER, o_customer_sk INTEGER, weekday VARCHAR(32), date VARCHAR(32), store INTEGER);"
Lineitem = "CREATE TABLE IF NOT EXISTS Lineitem_{}(li_order_id INTEGER, li_product_id INTEGER, quantity INTEGER, price DOUBLE);"
Product = "CREATE TABLE IF NOT EXISTS Product_{}(p_product_id INTEGER, name VARCHAR(32), department VARCHAR(32));"

# uc04 + uc07
Review = "CREATE TABLE IF NOT EXISTS Review_{}(ID INTEGER, text VARCHAR(65535));"
Product_Rating = "CREATE TABLE IF NOT EXISTS Product_Rating_{}(userID INTEGER, productID INTEGER);"

# uc06
Failures = "CREATE TABLE IF NOT EXISTS Failures_{}(date VARCHAR(32), serial_number INTEGER, model VARCHAR(32), smart_5_raw DOUBLE, smart_10_raw DOUBLE, smart_184_raw DOUBLE, smart_187_raw DOUBLE, smart_188_raw DOUBLE, smart_197_raw DOUBLE, smart_198_raw DOUBLE);"

# uc10
Financial_Account = "CREATE TABLE IF NOT EXISTS Financial_Account_{}(fa_customer_sk INTEGER, transaction_limit DOUBLE);"
Financial_Transactions = "CREATE TABLE IF NOT EXISTS Financial_Transactions_{}(amount DOUBLE, IBAN VARCHAR(32), senderID INTEGER, receiverID VARCHAR(32), transactionID VARCHAR(32), time VARCHAR(32));"

# load uc01
load_Order_Returns = '''LOAD DATA
/*+ PARALLEL(32) */
INFILE '{0}/tpcxai_datasets/{1}/serving/order_returns.csv'
INTO TABLE Order_Returns_{1}
FIELDS TERMINATED BY ',' ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;'''

# load uc03
load_Order_o = '''LOAD DATA
/*+ PARALLEL(32) */
INFILE '{0}/tpcxai_datasets/{1}/serving/order.csv'
INTO TABLE Order_o_{1}
FIELDS TERMINATED BY ',' ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;'''

load_Lineitem = '''LOAD DATA
/*+ PARALLEL(32) */
INFILE '{0}/tpcxai_datasets/{1}/serving/lineitem.csv'
INTO TABLE Lineitem_{1}
FIELDS TERMINATED BY ',' ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;'''

load_Product = '''LOAD DATA
/*+ PARALLEL(32) */
INFILE '{0}/tpcxai_datasets/{1}/serving/product.csv'
INTO TABLE Product_{1}
FIELDS TERMINATED BY ',' ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;'''

# load uc04 + uc07
load_Review = '''LOAD DATA
/*+ PARALLEL(32) */
INFILE '{0}/tpcxai_datasets/{1}/serving/Review.psv'
INTO TABLE Review_{1}
FIELDS TERMINATED BY '|' ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;'''

load_Product_Rating = '''LOAD DATA
/*+ PARALLEL(32) */
INFILE '{0}/tpcxai_datasets/{1}/serving/ProductRating.csv'
INTO TABLE Product_Rating_{1}
FIELDS TERMINATED BY ',' ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;'''

# load uc06
load_Failures = '''LOAD DATA
/*+ PARALLEL(32) */
INFILE '{0}/tpcxai_datasets/{1}/serving/failures.csv'
INTO TABLE Failures_{1}
FIELDS TERMINATED BY ',' ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;'''

# load uc10
load_Financial_Account = '''LOAD DATA
/*+ PARALLEL(32) */
INFILE '{0}/tpcxai_datasets/{1}/serving/financial_account.csv'
INTO TABLE Financial_Account_{1}
FIELDS TERMINATED BY ',' ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;'''

load_Financial_Transactions = '''LOAD DATA
/*+ PARALLEL(32) */
INFILE '{0}/tpcxai_datasets/{1}/serving/financial_transactions.csv'
INTO TABLE Financial_Transactions_{1}
FIELDS TERMINATED BY ',' ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;'''

def set_query_raven(test_sql):
	# create table
	'''test_sql.append(Expedia_S_listings.format(""))
	test_sql.append(Expedia_S_listings.format("_1G"))
	test_sql.append(Expedia_R1_hotels)
	test_sql.append(Expedia_R2_searches)
	test_sql.append(Flights_S_routes.format(""))
	test_sql.append(Flights_S_routes.format("_1G"))
	test_sql.append(Flights_R1_airlines)
	test_sql.append(Flights_R2_sairports)
	test_sql.append(Flights_R3_dairports)
	test_sql.append(LengthOfStay.format(""))
	test_sql.append(LengthOfStay.format("_1G"))
	test_sql.append(Credit_Card.format(""))
	test_sql.append(Credit_Card.format("_1G"))'''
	for i in range(10, 60, 10):
		size = "_{}G".format(i)
		#test_sql.append(Expedia_S_listings.format(size))
		test_sql.append(Flights_S_routes.format(size))
		#test_sql.append(LengthOfStay.format(size))
		#test_sql.append(Credit_Card.format(size))
	# load data
	# test_sql.append(load_Expedia_S_listings.format(prefix, ""))
	# test_sql.append(load_Expedia_S_listings.format(prefix, "_1G"))
	# test_sql.append(load_Expedia_R1_hotels.format(prefix))
	# test_sql.append(load_Expedia_R2_searches.format(prefix))
	# test_sql.append(load_Flights_S_routes.format(prefix, ""))
	# test_sql.append(load_Flights_S_routes.format(prefix, "_1G"))
	# test_sql.append(load_Flights_R1_airlines.format(prefix))
	# test_sql.append(load_Flights_R2_sairports.format(prefix))
	# test_sql.append(load_Flights_R3_dairports.format(prefix))
	# test_sql.append(load_LengthOfStay.format(prefix, ""))
	# test_sql.append(load_LengthOfStay.format(prefix, "_1G"))
	# test_sql.append(load_Credit_Card.format(prefix, ""))
	# test_sql.append(load_Credit_Card.format(prefix, "_1G"))
	for i in range(10, 60, 10):
		size = "_{}G".format(i)
		#test_sql.append(load_Expedia_S_listings.format(prefix, size))
		test_sql.append(load_Flights_S_routes.format(prefix, size))
		#test_sql.append(load_LengthOfStay.format(prefix, size))
		#test_sql.append(load_Credit_Card.format(prefix, size))


create_tpcx_ai_tables = [Order_Returns, Order_o, Lineitem, Product, Review, Product_Rating, Failures, Financial_Account, Financial_Transactions]
create_tpcx_ai_tables_sf100 = [Order_o, Lineitem, Product, Review, Product_Rating, Financial_Account, Financial_Transactions]
load_tpcx_ai_tables = [load_Order_Returns, load_Order_o, load_Lineitem, load_Product, load_Review, load_Product_Rating, load_Failures, load_Financial_Account, load_Financial_Transactions]
load_tpcx_ai_tables_sf100 = [load_Order_o, load_Lineitem, load_Product, load_Review, load_Product_Rating, load_Financial_Account, load_Financial_Transactions]

def set_query_tpcx_ai(test_sql):
	# create table
	for table in create_tpcx_ai_tables_sf100:
		test_sql.append(table.format("sf100"))
	# load data
	for load in load_tpcx_ai_tables_sf100:
		test_sql.append(load.format(prefix, "sf100"))

load_table_raven = False
load_table_tpcx_ai = True

if load_table_raven:
	# raven connection
	conn = pymysql.connect(host="127.0.0.1", port=2881, user="root", passwd="ulAFVBT0D4GDfMizqJXJ", db="raven")
	cur = conn.cursor()

	try:
		test_sql = []
		time_stat = 0
		cur.execute(trace_on) # open trace
		set_query_raven(test_sql)
		for i in test_sql:
			time_consuming = run_sql(cur, i)
			df = pd.DataFrame({'query': [i], 'execute': [time_consuming], 'time': [time.asctime()]})
			df.to_csv(output_path, index=True, mode='a', header=None)

	finally:
		cur.close()
		conn.close()

if load_table_tpcx_ai:
	# tpcx_ai connection
	conn = pymysql.connect(host="127.0.0.1", port=2881, user="root", passwd="ulAFVBT0D4GDfMizqJXJ", db="tpcx_ai")
	cur = conn.cursor()

	try:
		test_sql = []
		time_stat = 0
		cur.execute(trace_on) # open trace
		set_query_tpcx_ai(test_sql)
		for i in test_sql:
			time_consuming = run_sql(cur, i)
			df = pd.DataFrame({'query': [i], 'execute': [time_consuming], 'time': [time.asctime()]})
			df.to_csv(output_path, index=True, mode='a', header=None)

	finally:
		cur.close()
		conn.close()