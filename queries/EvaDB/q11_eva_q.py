import evadb
import pandas as pd

cursor = evadb.connect().cursor()
name = "uc07"

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

cursor.query("use backend_data {drop view if exists temp_eva_uc07}").execute()

cursor.query("use backend_data {CREATE VIEW temp_eva_uc07 AS SELECT userID, productID FROM Product_Rating GROUP BY userID, productID}").execute()

data = cursor.query("select userID, productID, uc07(userID, productID) from backend_data.temp_eva_uc07;").df()

print(data)

data.rename(columns={'uc07': 'score'})

# rank & window function
def rank(data: pd.DataFrame, n: int) -> pd.DataFrame:
    user_recommendations = []
    users = data.userid.unique()
    items = data.productid.unique()
    for u in users:
        ratings = data[data['userid'] == u].values.tolist()
        ratings = sorted(ratings, key=lambda t: t[2], reverse=True)[:n]
        for i in range(len(ratings)):
            ratings[i].append(i+1)
        user_recommendations.extend(ratings)
    return pd.DataFrame(user_recommendations, columns=['userID', 'productID', 'score', 'rank'])

user_recommendations = rank(data, 10)
print(user_recommendations)

    


