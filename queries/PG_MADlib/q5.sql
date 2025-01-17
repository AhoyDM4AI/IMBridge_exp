CREATE TEMP TABLE tpchq10_t as
select o_orderdate, c_name, n_name, c_address, c_phone, c_comment, c_custkey,
c_acctbal, o_totalprice, l_quantity, l_extendedprice, l_discount, l_tax,
         o_orderstatus, o_orderpriority, l_linestatus,
           l_shipinstruct, l_shipmode, n_nationkey, n_regionkey 
from customer, orders, lineitem, nation
where c_custkey = o_custkey and
l_orderkey = o_orderkey and
c_nationkey = n_nationkey;

ALTER TABLE tpchq10_t ADD COLUMN id SERIAL;

create TEMP TABLE tpchq10_t_feats as 
select id, c_acctbal/c_acctbal_std c_acctbal, o_totalprice/o_totalprice_std o_totalprice,
 l_quantity/l_quantity_std l_quantity, l_extendedprice/l_extendedprice_std l_extendedprice,
l_discount/l_discount_std l_discount, l_tax/l_tax_std l_tax,
o_orderstatus, o_orderpriority, l_linestatus, 
l_shipinstruct, l_shipmode, n_nationkey, n_regionkey 
from (select id, c_acctbal - c_acctbal_avg c_acctbal, o_totalprice - o_totalprice_avg o_totalprice,
l_quantity - l_quantity_avg l_quantity, l_extendedprice - l_extendedprice_avg l_extendedprice,
l_discount - l_discount_avg l_discount, l_tax - l_tax_avg l_tax,
o_orderstatus, o_orderpriority, l_linestatus,
l_shipinstruct, l_shipmode, n_nationkey, n_regionkey 
from tpchq10_t cross join tpchq10_avgs) t1 
cross join tpchq10_stds;

drop table if exists tpchq10_t_feats_out;
SELECT madlib.encode_categorical_variables('tpchq10_t_feats', 'tpchq10_t_feats_out',
'o_orderstatus, o_orderpriority, l_linestatus,
l_shipinstruct, l_shipmode, n_nationkey, n_regionkey', 
NULL,
'id, c_acctbal, o_totalprice, l_quantity, l_extendedprice, l_discount, l_tax');

drop table if exists tpchq10_t_out;
SELECT madlib.forest_predict('tpchq10_rf_model',        -- tree model
                             'tpchq10_t_feats_out',             -- new data table
                             'tpchq10_t_out',  -- output table
                             'response');      -- show response

explain analyze
select 
c_custkey,
c_name,
sum(l_extendedprice * (1 - l_discount)) as revenue,
c_acctbal,
n_name,
c_address,
c_phone,
c_comment
from tpchq10_t JOIN tpchq10_t_out USING (id) 
where estimated_l_returnflag = 'R' and
o_orderdate >= DATE'1993-10-01' and 
o_orderdate < DATE'1993-10-01' + interval '3' month 
group by c_custkey, c_name, c_acctbal, c_phone, n_name, c_address, c_comment
order by revenue desc;
