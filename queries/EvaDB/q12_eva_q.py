import evadb
import pandas as pd

cursor = evadb.connect().cursor()
name = "uc10"

params = {
    "user": "root",
    "host": "49.52.27.23",
    "port": "4001",
    "database": "tpcx_ai",
    "password": "oGslD19GXXy6F5bhzzox"
}
query = f"CREATE DATABASE backend_data WITH ENGINE='mysql', PARAMETERS={params};"
cursor.query(query).df()

cursor.query(f"CREATE FUNCTION IF NOT EXISTS {name} IMPL './{name}_eva.py'").execute()

cursor.query("use backend_data {drop view if exists temp_eva_uc10}").execute()

cursor.query("use backend_data {CREATE VIEW temp_eva_uc10 AS SELECT transactionID, CAST(amount/transaction_limit AS DOUBLE) amount_norm, CAST(HOUR(STR_TO_DATE(time, '%Y-%m-%dT%H:%M'))/23 AS DOUBLE) business_hour_norm from Financial_Account join Financial_Transactions on Financial_Account.fa_customer_sk = Financial_Transactions.senderID}").execute()

data = cursor.query("select transactionID, uc10(business_hour_norm, amount_norm) from backend_data.temp_eva_uc10;").df()

print(data)
