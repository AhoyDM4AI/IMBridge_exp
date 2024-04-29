import evadb

cursor = evadb.connect().cursor()
name = "uc06"

params = {
    "user": "root",
    "host": "xxx",
    "port": "2881",
    "database": "tpcx_ai",
    "password": "oGslD19GXXy6F5bhzzox"
}
query = f"CREATE DATABASE backend_data WITH ENGINE='mysql', PARAMETERS={params};"
cursor.query(query).df()

cursor.query(f"CREATE FUNCTION IF NOT EXISTS {name} IMPL './{name}_eva.py'").execute()

cursor.query("use backend_data {drop view if exists temp_eva_uc06}").execute()

cursor.query("use backend_data {create view tb_eva_uc06 as select serial_number, smart_5_raw, smart_10_raw, smart_184_raw, smart_187_raw, smart_188_raw, smart_197_raw, smart_198_raw from Failures}").execute()

print(cursor.query("select serial_number, uc06(smart_5_raw, smart_10_raw, smart_184_raw, smart_187_raw, smart_188_raw, smart_197_raw, smart_198_raw) from backend_data.tb_eva_uc06;").df())
