import evadb

cursor = evadb.connect().cursor()
name = "pf8"

params = {
    "user": "xxx",
    "host": "49.52.27.23",
    "port": "2881",
    "database": "raven",
    "password": "oGslD19GXXy6F5bhzzox",
}
query = f"CREATE DATABASE backend_data WITH ENGINE='mysql', PARAMETERS={params};"
cursor.query(query).df()

cursor.query(f"CREATE FUNCTION IF NOT EXISTS {name} IMPL './{name}_eva.py'").execute()

cursor.query("use backend_data {drop view if exists temp_eva_pf7}").execute()

#cursor.query("use backend_data {CREATE VIEW temp_eva_pf7 AS SELECT }").execute()

print(cursor.query("SELECT Time, Amount, pf8(V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11, V12, V13, V14, V15, V16, V17, V18, V19, V20, V21, V22, V23, V24, V25, V26, V27, V28, Amount) FROM backend_data.Credit_Card_eva WHERE V1 > 1 AND V2 < 0.27 AND V3 > 0.3;").df())
