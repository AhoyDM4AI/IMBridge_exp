-- Select tpcx uc01 Staged Prediction
-- Hash Join
select /*+NO_REWRITE*/ ratio_tbl.o_customer_sk, PREDICT tpcx_ai_uc01_staged(CAST(COALESCE(return_ratio,0) AS DOUBLE), CAST(COALESCE(frequency,0) AS DOUBLE)) as cluster
from (select o_customer_sk, avg(ratio) return_ratio 
from (select o_customer_sk, return_row_price_sum/row_price_sum ratio
from (select o_order_id, o_customer_sk, SUM(row_price) row_price_sum, SUM(return_row_price) return_row_price_sum, SUM(invoice_year) invoice_year_min 
from (select o_order_id, o_customer_sk, extract(year FROM cast(date0 as DATE)) invoice_year, quantity*price row_price, or_return_quantity*price return_row_price 
from (select o_order_id, o_customer_sk, date date0, li_product_id, price, quantity, or_return_quantity  
from (select * from Lineitem_sf10 left join Order_Returns_sf10 
on Lineitem_sf10.li_order_id = Order_Returns_sf10.or_order_id 
and Lineitem_sf10.li_product_id = Order_Returns_sf10.or_product_id
) returns_data Join Order_o_sf10 on returns_data.li_order_id=Order_o_sf10.o_order_id)) 
group by o_order_id, o_customer_sk)
)
group by o_customer_sk
) ratio_tbl 
join (select o_customer_sk, avg(o_order_id) frequency
from (select o_customer_sk, invoice_year_min, count(DISTINCT o_order_id) o_order_id
from (select o_order_id, o_customer_sk, SUM(row_price) row_price_sum, SUM(return_row_price) return_row_price_sum, SUM(invoice_year) invoice_year_min 
from (select o_order_id, o_customer_sk, extract(year FROM cast(date0 as DATE)) invoice_year, quantity*price row_price, or_return_quantity*price return_row_price 
from (select o_order_id, o_customer_sk, date date0, li_product_id, price, quantity, or_return_quantity  
from (select * from Lineitem_sf10 left join Order_Returns_sf10 
on Lineitem_sf10.li_order_id = Order_Returns_sf10.or_order_id 
and Lineitem_sf10.li_product_id = Order_Returns_sf10.or_product_id
) returns_data Join Order_o_sf10 on returns_data.li_order_id=Order_o_sf10.o_order_id)) 
group by o_order_id, o_customer_sk
)
group by o_customer_sk, invoice_year_min
) group by o_customer_sk
) frequency_tbl on ratio_tbl.o_customer_sk = frequency_tbl.o_customer_sk;