CREATE Temp TABLE tpchq5_t as
select n_name, r_name, o_orderdate,
c_acctbal, o_totalprice, l_quantity, l_extendedprice,
l_discount, l_tax, s_acctbal,
o_orderstatus, l_returnflag, l_linestatus,
l_shipinstruct, l_shipmode, n_nationkey, n_regionkey 
from
 customer,
 orders,
 lineitem,
 supplier,
 nation,
 region
where
 c_custkey = o_custkey
 and l_orderkey = o_orderkey
 and l_suppkey = s_suppkey
 and c_nationkey = s_nationkey
 and s_nationkey = n_nationkey
 and n_regionkey = r_regionkey;

 ALTER TABLE tpchq5_t ADD COLUMN id SERIAL;

create TEMP TABLE tpchq5_t_feats as
select n_name, r_name, o_orderdate, c_acctbal/c_acctbal_std c_acctbal, o_totalprice/o_totalprice_std o_totalprice,
 l_quantity/l_quantity_std l_quantity, l_extendedprice/l_extendedprice_std l_extendedprice,
l_discount/l_discount_std l_discount, l_tax/l_tax_std l_tax, s_acctbal/s_acctbal_std s_acctbal,
o_orderstatus, l_returnflag, l_linestatus, 
l_shipinstruct, l_shipmode, n_nationkey, n_regionkey
from (select n_name, r_name, o_orderdate, c_acctbal - c_acctbal_avg c_acctbal,
 o_totalprice - o_totalprice_avg o_totalprice,
l_quantity - l_quantity_avg l_quantity, l_extendedprice - l_extendedprice_avg l_extendedprice,
l_discount - l_discount_avg l_discount, l_tax - l_tax_avg l_tax, s_acctbal - s_acctbal_avg s_acctbal,
o_orderstatus, l_returnflag, l_linestatus,
l_shipinstruct, l_shipmode, n_nationkey, n_regionkey 
from tpchq5_t cross join tpchq5_avgs) t1 
cross join tpchq5_stds;

drop table if exists tpchq5_t_feats_out;
SELECT madlib.encode_categorical_variables('tpchq5_t_feats', 'tpchq5_t_feats_out',
'o_orderstatus, l_returnflag, l_linestatus,
l_shipinstruct, l_shipmode, n_nationkey, n_regionkey', 
NULL,
'c_acctbal, o_totalprice, l_quantity, l_extendedprice,
l_discount, l_tax, s_acctbal, n_name, r_name, o_orderdate');

-- Add the id column for prediction function
ALTER TABLE tpchq5_t_feats_out ADD COLUMN id SERIAL;

-- Predict probabilities for all categories using the original data
drop table if exists tpchq5_t_out;
SELECT madlib.mlp_predict('tpchq5_mlp','tpchq5_t_feats_out', 'id', 'tpchq5_t_out', 'response');

explain analyze 
select
 n_name,
 sum(l_extendedprice * (1 - l_discount)) as revenue
from  tpchq5_t JOIN tpchq5_t_out USING (id) 
where estimated_o_orderpriority = '1' 
 and r_name = 'ASIA'
 and o_orderdate >= date '1994-01-01'
 and o_orderdate < date '1994-01-01' + interval '1' year 
group by n_name order by
revenue desc;

