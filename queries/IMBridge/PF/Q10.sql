-- Select tpcx uc03 Staged Prediction
select store, department, PREDICT tpcx_ai_uc03(store, department) 
from (select store, department
from Order_o_sf10 Join Lineitem_sf10 on Order_o_sf10.o_order_id = Lineitem_sf10.li_order_id
Join Product_sf10 on Lineitem_sf10.li_product_id = Product_sf10.p_product_id 
group by store, department);