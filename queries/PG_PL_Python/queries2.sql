SET statement_timeout = 10800000;

--q7
SELECT 7;
EXPLAIN ANALYZE select ratio_tbl.o_customer_sk, udf7(cast(COALESCE(return_ratio,0.0) as REAL), cast(COALESCE(frequency,0.0) as REAL))
from (select o_customer_sk, avg(ratio) return_ratio
from (select o_customer_sk, return_row_price_sum/row_price_sum ratio
from (select o_order_id, o_customer_sk, SUM(row_price) row_price_sum, SUM(return_row_price) return_row_price_sum, SUM(invoice_year) invoice_year_min
from (select o_order_id, o_customer_sk, extract('year' FROM cast(date0 as DATE)) invoice_year, quantity*price row_price, or_return_quantity*price return_row_price
from (select o_order_id, o_customer_sk, date date0, li_product_id, price, quantity, or_return_quantity
from (select * from lineitem left join Order_Returns
on lineitem.li_order_id = Order_Returns.or_order_id
and lineitem.li_product_id = Order_Returns.or_product_id
) returns_data Join Order_o on returns_data.li_order_id=Order_o.o_order_id) t4
) t3
group by o_order_id, o_customer_sk) t2
) t1
group by o_customer_sk
) ratio_tbl
join (select o_customer_sk, avg(o_order_id) frequency
from (select o_customer_sk, invoice_year_min, count(DISTINCT o_order_id) o_order_id
from (select o_order_id, o_customer_sk, SUM(row_price) row_price_sum, SUM(return_row_price) return_row_price_sum, SUM(invoice_year) invoice_year_min
from (select o_order_id, o_customer_sk, extract('year' FROM cast(date0 as DATE)) invoice_year, quantity*price row_price, or_return_quantity*price return_row_price
from (select o_order_id, o_customer_sk, date date0, li_product_id, price, quantity, or_return_quantity
from (select * from lineitem left join Order_Returns
on lineitem.li_order_id = Order_Returns.or_order_id
and lineitem.li_product_id = Order_Returns.or_product_id
) returns_data Join Order_o on returns_data.li_order_id=Order_o.o_order_id) t8
) t7
group by o_order_id, o_customer_sk
) t6
group by o_customer_sk, invoice_year_min
) t5 group by o_customer_sk
) frequency_tbl on ratio_tbl.o_customer_sk = frequency_tbl.o_customer_sk;

--q8
SELECT 8;
EXPLAIN ANALYZE select store, department, udf8(cast(store as text), cast(department as text))
from (select store, department
from (Order_o Join Lineitem on Order_o.o_order_id = Lineitem.li_order_id) data
Join Product on data.li_product_id=Product.p_product_id
group by store,department) t1;

--q9
SELECT 9;
EXPLAIN ANALYZE select udf9(cast(txt as text)) from
(select DISTINCT text txt from Review) t1;

--q10
SELECT 10;
EXPLAIN ANALYZE select serial_number, udf10(smart_5_raw, smart_10_raw, smart_184_raw, smart_187_raw, smart_188_raw, smart_197_raw, smart_198_raw)
from Failures;

--q11
SELECT 11;
EXPLAIN ANALYZE select userID, productID, r, score
from (select userID, productID, score, rank() OVER (PARTITION BY userID ORDER BY score) as r
from (select userID, productID, udf11(userID, productID) score
from (select userID, productID
from Product_Rating
group by userID, productID) t3
) t2
) t1
where r <=10;

--q12
SELECT 12;
EXPLAIN ANALYZE select transactionID, udf12(amount_norm, business_hour_norm)
from (select transactionID, amount/transaction_limit amount_norm, extract(hour from to_timestamp(time, '%Y-%m-%dT%H:%M'))/23 business_hour_norm
from Financial_Account join Financial_Transactions on Financial_Account.fa_customer_sk = Financial_Transactions.senderID) t1;