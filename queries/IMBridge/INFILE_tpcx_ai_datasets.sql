SET GLOBAL SECURE_FILE_PRIV = '';

GRANT FILE ON *.* TO root;

set global ob_trx_idle_timeout=36000000000;
set global ob_trx_timeout=36000000000;
set global ob_query_timeout=36000000000;

CREATE DATABASE tpcx_ai;

USE tpcx_ai;

CREATE TABLE Order_Returns_sf10(or_order_id INTEGER, or_product_id INTEGER, or_return_quantity INTEGER);

LOAD DATA
/*+ PARALLEL(16) */
INFILE '{HOME_PATH}/workload/tpcxai_datasets/sf10/serving/order_returns.csv'
INTO TABLE Order_Returns_sf10
FIELDS TERMINATED BY ',' ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;

CREATE TABLE Order_o_sf10(o_order_id INTEGER, o_customer_sk INTEGER, weekday VARCHAR(32), date VARCHAR(32), store INTEGER);

CREATE TABLE Lineitem_sf10(li_order_id INTEGER, li_product_id INTEGER, quantity INTEGER, price DOUBLE);

CREATE TABLE Product_sf10(p_product_id INTEGER, name VARCHAR(32), department VARCHAR(32));

LOAD DATA
/*+ PARALLEL(16) */
INFILE '{HOME_PATH}/workload/tpcxai_datasets/sf10/serving/order.csv'
INTO TABLE Order_o_sf10
FIELDS TERMINATED BY ',' ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;

LOAD DATA
/*+ PARALLEL(16) */
INFILE '{HOME_PATH}/workload/tpcxai_datasets/sf10/serving/product.csv'
INTO TABLE Product_sf10
FIELDS TERMINATED BY ',' ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;

LOAD DATA
/*+ PARALLEL(16) */
INFILE '{HOME_PATH}/workload/tpcxai_datasets/sf10/serving/lineitem.csv'
INTO TABLE Lineitem_sf10
FIELDS TERMINATED BY ',' ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;

CREATE TABLE Review_sf10(ID INTEGER, text VARCHAR(65535));
CREATE TABLE Product_Rating_sf10(userID INTEGER, productID INTEGER);

LOAD DATA
INFILE '{HOME_PATH}/workload/tpcxai_datasets/sf10/serving/Review.psv'
INTO TABLE Review_sf10
FIELDS TERMINATED BY '|' ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;

LOAD DATA
INFILE '{HOME_PATH}/workload/tpcxai_datasets/sf10/serving/ProductRating.csv'
INTO TABLE Product_Rating_sf10
FIELDS TERMINATED BY ',' ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;

CREATE TABLE Failures_sf10(date VARCHAR(32), serial_number INTEGER, model VARCHAR(32), smart_5_raw DOUBLE, smart_10_raw DOUBLE, 
smart_184_raw DOUBLE, smart_187_raw DOUBLE, smart_188_raw DOUBLE, smart_197_raw DOUBLE, smart_198_raw DOUBLE);

LOAD DATA
/*+ PARALLEL(16) */
INFILE '{HOME_PATH}/workload/tpcxai_datasets/sf10/serving/failures.csv'
INTO TABLE Failures_sf10
FIELDS TERMINATED BY ',' ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;

CREATE TABLE Financial_Account_sf10(fa_customer_sk INTEGER, transaction_limit DOUBLE);

CREATE TABLE Financial_Transactions_sf10(amount DOUBLE, IBAN VARCHAR(32), senderID INTEGER, receiverID VARCHAR(32), transactionID VARCHAR(32), time VARCHAR(32));

LOAD DATA
/*+ PARALLEL(16) */
INFILE '{HOME_PATH}/workload/tpcxai_datasets/sf10/serving/financial_account.csv'
INTO TABLE Financial_Account_sf10
FIELDS TERMINATED BY ',' ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;

LOAD DATA
/*+ PARALLEL(16) */
INFILE '{HOME_PATH}/workload/tpcxai_datasets/sf10/serving/financial_transactions.csv'
INTO TABLE Financial_Transactions_sf10
FIELDS TERMINATED BY ',' ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;
