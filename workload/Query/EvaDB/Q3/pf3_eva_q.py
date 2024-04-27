import evadb

cursor = evadb.connect().cursor()
name = "pf3"

params = {
    "user": "root",
    "host": "xxx",
    "port": "2881",
    "database": "raven",
    "password": "oGslD19GXXy6F5bhzzox",
}
query = f"CREATE DATABASE backend_data WITH ENGINE='mysql', PARAMETERS={params};"
cursor.query(query).df()

cursor.query(f"CREATE FUNCTION IF NOT EXISTS {name} IMPL './{name}_eva.py'").execute()

cursor.query("use backend_data {drop view if exists temp_eva_pf3}").execute()

cursor.query("use backend_data {CREATE VIEW temp_eva_pf3 AS SELECT Flights_S_routes_extension.airlineid, Flights_S_routes_extension.sairportid, Flights_S_routes_extension.dairportid, slatitude, slongitude, dlatitude, dlongitude, name1, name2, name4, acountry, active, scity, scountry, stimezone, sdst, dcity, dcountry, dtimezone, ddst FROM Flights_S_routes_extension JOIN Flights_R1_airlines ON Flights_S_routes_extension.airlineid = Flights_R1_airlines.airlineid JOIN Flights_R2_sairports ON Flights_S_routes_extension.sairportid = Flights_R2_sairports.sairportid JOIN Flights_R3_dairports ON Flights_S_routes_extension.dairportid = Flights_R3_dairports.dairportid}").execute()

print(cursor.query("SELECT airlineid, sairportid, dairportid, pf3(slatitude, slongitude, dlatitude, dlongitude, name1, name2, name4, acountry, active, scity, scountry, stimezone, sdst, dcity, dcountry, dtimezone, ddst) FROM backend_data.temp_eva_pf3 WHERE name2 = 't' and name4 = 't' and name1 = 3;").df())
