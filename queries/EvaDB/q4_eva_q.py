import evadb

cursor = evadb.connect().cursor()
name = "pf5"

params = {
    "user": "root",
    "host": "49.52.27.23",
    "port": "3302",
    "database": "raven",
    "password": "ulAFVBT0D4GDfMizqJXJ",
}
query = f"CREATE DATABASE backend_data WITH ENGINE='mysql', PARAMETERS={params};"
cursor.query(query).df()

cursor.query(f"CREATE FUNCTION IF NOT EXISTS {name} IMPL './{name}_eva.py'").execute()

cursor.query("use backend_data {drop view if exists temp_eva_pf5}").execute()

cursor.query("use backend_data {CREATE VIEW temp_eva_pf5 AS SELECT * from LengthOfStay_1G }").execute()

print(cursor.query("SELECT eid, pf5(hematocrit, neutrophils, sodium, glucose, bloodureanitro, creatinine, bmi, pulse, respiration, secondarydiagnosisnonicd9, rcount, gender, dialysisrenalendstage, asthma, irondef, pneum, substancedependence, psychologicaldisordermajor, depress, psychother, fibrosisandother, malnutrition, hemo) FROM backend_data.temp_eva_pf5 WHERE hematocrit > 10 AND neutrophils > 10 AND bloodureanitro < 20 AND pulse < 70;").df())


