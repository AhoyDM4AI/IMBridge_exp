import duckdb
import joblib
import numpy as np
import pandas as pd
from duckdb.typing import BIGINT, DOUBLE, FLOAT, VARCHAR
from sklearn.linear_model import LogisticRegression

con = duckdb.connect("imbridge.db")

name = "q12"
mname = "uc10"
model_file_name = f"/home/model/{mname}/{mname}.python.model"

model = joblib.load(model_file_name)


def udf(business_hour_norm, amount_norm):
    data = pd.DataFrame({
        'business_hour_norm': business_hour_norm,
        'amount_norm': amount_norm
    })
    return model.predict(data)


con.create_function("udf", udf, [DOUBLE, DOUBLE], BIGINT, type="arrow")

# con.sql("SET threads TO 1;")

res = con.sql('''
explain analyze select transactionID, udf(amount_norm, business_hour_norm) 
from (select transactionID, amount/transaction_limit amount_norm, hour(strptime(time, '%Y-%m-%dT%H:%M'))/23 business_hour_norm 
from Financial_Account join Financial_Transactions on Financial_Account.fa_customer_sk = Financial_Transactions.senderID);
''')

print(name)
