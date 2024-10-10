import evadb

cursor = evadb.connect().cursor()
name = "uc01"

params = {
    "user": "root",
    "host": "49.52.27.23",
    "port": "3302",
    "database": "tpcx_ai",
    "password": "ulAFVBT0D4GDfMizqJXJ",
}
query = f"CREATE DATABASE backend_data WITH ENGINE='mysql', PARAMETERS={params};"
cursor.query(query).df()

cursor.query(f"CREATE FUNCTION IF NOT EXISTS uc01 IMPL './{name}_eva.py'").execute()

cursor.query("use backend_data {drop view if exists temp_eva_uc01}").execute()

cursor.query('''
use backend_data {
create view temp_eva_uc01 as
select ratio_tbl.o_customer_sk, CAST(COALESCE(return_ratio,0) AS DOUBLE) return_ratio, CAST(COALESCE(frequency,0) AS DOUBLE) frequency
from (select o_customer_sk, avg(ratio) return_ratio 
from (select o_customer_sk, return_row_price_sum/row_price_sum ratio
from (select o_order_id, o_customer_sk, SUM(row_price) row_price_sum, SUM(return_row_price) return_row_price_sum, SUM(invoice_year) invoice_year_min 
from (select o_order_id, o_customer_sk, extract(year FROM cast(date0 as DATE)) invoice_year, quantity*price row_price, or_return_quantity*price return_row_price 
from (select o_order_id, o_customer_sk, date date0, li_product_id, price, quantity, or_return_quantity  
from (select * from lineitem_sf10 left join Order_Returns_sf10 
on lineitem_sf10.li_order_id = Order_Returns_sf10.or_order_id 
and lineitem_sf10.li_product_id = Order_Returns_sf10.or_product_id
) returns_data_sf10 Join Order_o_sf10 on returns_data_sf10.li_order_id=Order_o_sf10.o_order_id)) 
group by o_order_id, o_customer_sk)
)
group by o_customer_sk
) ratio_tbl 
join (select o_customer_sk, avg(o_order_id) frequency
from (select o_customer_sk, invoice_year_min, count(DISTINCT o_order_id) o_order_id
from (select o_order_id, o_customer_sk, SUM(row_price) row_price_sum, SUM(return_row_price) return_row_price_sum, SUM(invoice_year) invoice_year_min 
from (select o_order_id, o_customer_sk, extract(year FROM cast(date0 as DATE)) invoice_year, quantity*price row_price, or_return_quantity*price return_row_price 
from (select o_order_id, o_customer_sk, date date0, li_product_id, price, quantity, or_return_quantity  
from (select * from lineitem_sf10 left join Order_Returns_sf10 
on lineitem_sf10.li_order_id = Order_Returns_sf10.or_order_id 
and lineitem_sf10.li_product_id = Order_Returns_sf10.or_product_id
) returns_data_sf10 Join Order_o_sf10 on returns_data_sf10.li_order_id=Order_o_sf10.o_order_id)) 
group by o_order_id, o_customer_sk
)
group by o_customer_sk, invoice_year_min
) group by o_customer_sk
) frequency_tbl on ratio_tbl.o_customer_sk = frequency_tbl.o_customer_sk}
''').execute()

#cursor.query("create table tb_eva as select * from backend_data.temp_eva_uc01;").execute()

print(cursor.query("select o_customer_sk, uc01(return_ratio, frequency) from backend_data.temp_eva_uc01;").df())
