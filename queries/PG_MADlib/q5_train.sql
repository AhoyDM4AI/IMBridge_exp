CREATE TEMP TABLE tpchq10_raw as
select
c_acctbal, o_totalprice, l_quantity, l_extendedprice, l_discount, l_tax,
         o_orderstatus, o_orderpriority, l_linestatus,
           l_shipinstruct, l_shipmode, n_nationkey, n_regionkey, l_returnflag 
from customer, orders, lineitem, nation
where c_custkey = o_custkey and
l_orderkey = o_orderkey and
c_nationkey = n_nationkey;

ALTER TABLE tpchq10_raw ADD COLUMN id SERIAL;

CREATE TABLE tpchq10_stds as
SELECT STDDEV(c_acctbal) c_acctbal_std, STDDEV(o_totalprice) o_totalprice_std, STDDEV(l_quantity) l_quantity_std,
 STDDEV(l_extendedprice) l_extendedprice_std, STDDEV(l_discount) l_discount_std, STDDEV(l_tax) l_tax_std 
FROM tpchq10_raw;

CREATE TABLE tpchq10_avgs as
SELECT AVG(c_acctbal) c_acctbal_avg, AVG(o_totalprice) o_totalprice_avg, AVG(l_quantity) l_quantity_avg,
 AVG(l_extendedprice) l_extendedprice_avg, AVG(l_discount) l_discount_avg, AVG(l_tax) l_tax_avg 
FROM tpchq10_raw;

create TEMP TABLE tpchq10_feats as 
select id, c_acctbal/c_acctbal_std c_acctbal, o_totalprice/o_totalprice_std o_totalprice,
 l_quantity/l_quantity_std l_quantity, l_extendedprice/l_extendedprice_std l_extendedprice,
l_discount/l_discount_std l_discount, l_tax/l_tax_std l_tax,
o_orderstatus, o_orderpriority, l_linestatus, 
l_shipinstruct, l_shipmode, n_nationkey, n_regionkey, l_returnflag
from (select id, c_acctbal - c_acctbal_avg c_acctbal, o_totalprice - o_totalprice_avg o_totalprice,
l_quantity - l_quantity_avg l_quantity, l_extendedprice - l_extendedprice_avg l_extendedprice,
l_discount - l_discount_avg l_discount, l_tax - l_tax_avg l_tax,
o_orderstatus, o_orderpriority, l_linestatus,
l_shipinstruct, l_shipmode, n_nationkey, n_regionkey, l_returnflag 
from tpchq10_raw cross join tpchq10_avgs) t1 
cross join tpchq10_stds;

SELECT madlib.encode_categorical_variables('tpchq10_feats', 'tpchq10_feats_out',
'o_orderstatus, o_orderpriority, l_linestatus,
l_shipinstruct, l_shipmode, n_nationkey, n_regionkey', 
NULL,
'id, c_acctbal, o_totalprice, l_quantity, l_extendedprice, l_discount, l_tax, l_returnflag');

SELECT madlib.forest_train('tpchq10_feats_out',         -- source table
'tpchq10_rf_model',    -- output model table
'id',              -- id column
'l_returnflag',           -- response
'"c_acctbal","o_totalprice","l_quantity","l_extendedprice","l_discount","l_tax","o_orderstatus_F","o_orderstatus_O","o_orderstatus_P","o_orderpriority_1-URGENT","o_orderpriority_2-HIGH","o_orderpriority_3-MEDIUM","o_orderpriority_4-NOT SPECIFIED","o_orderpriority_5-LOW","l_linestatus_F","l_linestatus_O","l_shipinstruct_COLLECT COD","l_shipinstruct_DELIVER IN PERSON","l_shipinstruct_NONE","l_shipinstruct_TAKE BACK RETURN","l_shipmode_AIR","l_shipmode_FOB","l_shipmode_MAIL","l_shipmode_RAIL","l_shipmode_REG AIR","l_shipmode_SHIP","l_shipmode_TRUCK","n_nationkey_0","n_nationkey_1","n_nationkey_10","n_nationkey_11","n_nationkey_12","n_nationkey_13","n_nationkey_14","n_nationkey_15","n_nationkey_16","n_nationkey_17","n_nationkey_18","n_nationkey_19","n_nationkey_2","n_nationkey_20","n_nationkey_21","n_nationkey_22","n_nationkey_23","n_nationkey_24","n_nationkey_3","n_nationkey_4","n_nationkey_5","n_nationkey_6","n_nationkey_7","n_nationkey_8","n_nationkey_9","n_regionkey_0","n_regionkey_1","n_regionkey_2","n_regionkey_3","n_regionkey_4"',   -- features
NULL,              -- exclude columns
NULL,              -- grouping columns
8::integer,       -- number of trees
2::integer,        -- number of random features
TRUE::boolean,     -- variable importance
1::integer,        -- num_permutations
8::integer,        -- max depth
3::integer,        -- min split
1::integer,        -- min bucket
10::integer,        -- number of splits per continuous variable
NULL,
true
);