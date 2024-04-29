import evadb

cursor = evadb.connect().cursor()
name = "pf6"

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

cursor.query("use backend_data {drop view if exists temp_eva_pf6}").execute()

#cursor.query("use backend_data {CREATE VIEW temp_eva_pf6 AS SELECT }").execute()

print(cursor.query("SELECT eid, pf6(hematocrit, neutrophils, sodium, glucose, bloodureanitro, creatinine, bmi, pulse, respiration, secondarydiagnosisnonicd9, rcount, gender, dialysisrenalendstage, asthma, irondef, pneum, substancedependence, psychologicaldisordermajor, depress, psychother, fibrosisandother, malnutrition, hemo) FROM backend_data.LengthOfStay_eva WHERE hematocrit > 10 AND neutrophils > 10 AND bloodureanitro < 20 AND pulse < 70;").df())
