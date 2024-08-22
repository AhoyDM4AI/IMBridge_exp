import duckdb
import joblib
import numpy as np
import pandas as pd
from duckdb.typing import BIGINT, DOUBLE, FLOAT
# K-Means clustering
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

con = duckdb.connect("imbridge.db")

name = "uc01"
model_file_name = f"./model/{name}/{name}.python.model"

model = joblib.load(model_file_name)


def udf(return_ratio, frequency):
    feat = pd.DataFrame({
        'return_ratio': return_ratio,
        'frequency': frequency
    })
    return model.predict(feat)


con.create_function("udf", udf, [DOUBLE, DOUBLE], BIGINT, type="arrow")

#con.sql("SET threads TO 1;")

res = con.sql('''
explain analyze select ratio_tbl.o_customer_sk, udf(COALESCE(return_ratio,0), COALESCE(frequency,0))
from (select o_customer_sk, mean(ratio) return_ratio
from (select o_customer_sk, return_row_price_sum/row_price_sum ratio
from (select o_order_id, o_customer_sk, SUM(row_price) row_price_sum, SUM(return_row_price) return_row_price_sum, SUM(invoice_year) invoice_year_min
from (select o_order_id, o_customer_sk, extract('year' FROM cast(date0 as DATE)) invoice_year, quantity*price row_price, or_return_quantity*price return_row_price
from (select o_order_id, o_customer_sk, date date0, li_product_id, price, quantity, or_return_quantity
from (select * from lineitem left join Order_Returns
on lineitem.li_order_id = Order_Returns.or_order_id
and lineitem.li_product_id = Order_Returns.or_product_id
) returns_data Join Order_o on returns_data.li_order_id=Order_o.o_order_id))
group by o_order_id, o_customer_sk)
)
group by o_customer_sk
) ratio_tbl
join (select o_customer_sk, mean(o_order_id) frequency
from (select o_customer_sk, invoice_year_min, count(DISTINCT o_order_id) o_order_id
from (select o_order_id, o_customer_sk, SUM(row_price) row_price_sum, SUM(return_row_price) return_row_price_sum, SUM(invoice_year) invoice_year_min
from (select o_order_id, o_customer_sk, extract('year' FROM cast(date0 as DATE)) invoice_year, quantity*price row_price, or_return_quantity*price return_row_price
from (select o_order_id, o_customer_sk, date date0, li_product_id, price, quantity, or_return_quantity
from (select * from lineitem left join Order_Returns
on lineitem.li_order_id = Order_Returns.or_order_id
and lineitem.li_product_id = Order_Returns.or_product_id
) returns_data Join Order_o on returns_data.li_order_id=Order_o.o_order_id))
group by o_order_id, o_customer_sk
)
group by o_customer_sk, invoice_year_min
) group by o_customer_sk
) frequency_tbl on ratio_tbl.o_customer_sk = frequency_tbl.o_customer_sk;
''')

print(name)