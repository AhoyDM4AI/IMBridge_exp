CREATE TABLE tpchq5_raw as
select
c_acctbal, o_totalprice, l_quantity, l_extendedprice,
l_discount, l_tax, s_acctbal,
o_orderstatus, l_returnflag, l_linestatus,
l_shipinstruct, l_shipmode, n_nationkey, n_regionkey,  o_orderpriority
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

CREATE TABLE tpchq5_stds as
SELECT STDDEV(c_acctbal) c_acctbal_std, STDDEV(o_totalprice) o_totalprice_std, STDDEV(l_quantity) l_quantity_std,
 STDDEV(l_extendedprice) l_extendedprice_std, STDDEV(l_discount) l_discount_std, STDDEV(l_tax) l_tax_std, STDDEV(s_acctbal) s_acctbal_std
FROM tpchq5_raw;

CREATE TABLE tpchq5_avgs as
SELECT AVG(c_acctbal) c_acctbal_avg, AVG(o_totalprice) o_totalprice_avg, AVG(l_quantity) l_quantity_avg,
 AVG(l_extendedprice) l_extendedprice_avg, AVG(l_discount) l_discount_avg, AVG(l_tax) l_tax_avg, AVG(s_acctbal) s_acctbal_avg 
FROM tpchq5_raw;

create TEMP TABLE tpchq5_feats as 
select c_acctbal/c_acctbal_std c_acctbal, o_totalprice/o_totalprice_std o_totalprice,
 l_quantity/l_quantity_std l_quantity, l_extendedprice/l_extendedprice_std l_extendedprice,
l_discount/l_discount_std l_discount, l_tax/l_tax_std l_tax, s_acctbal/s_acctbal_std s_acctbal,
o_orderstatus, l_returnflag, l_linestatus, 
l_shipinstruct, l_shipmode, n_nationkey, n_regionkey,  o_orderpriority
from (select c_acctbal - c_acctbal_avg c_acctbal, o_totalprice - o_totalprice_avg o_totalprice,
l_quantity - l_quantity_avg l_quantity, l_extendedprice - l_extendedprice_avg l_extendedprice,
l_discount - l_discount_avg l_discount, l_tax - l_tax_avg l_tax, s_acctbal - s_acctbal_avg s_acctbal,
o_orderstatus, l_returnflag, l_linestatus,
l_shipinstruct, l_shipmode, n_nationkey, n_regionkey,  o_orderpriority 
from tpchq5_raw cross join tpchq5_avgs) t1 
cross join tpchq5_stds;

SELECT madlib.encode_categorical_variables('tpchq5_feats', 'tpchq5_feats_out',
'o_orderstatus, l_returnflag, l_linestatus,
l_shipinstruct, l_shipmode, n_nationkey, n_regionkey', 
NULL,
'c_acctbal, o_totalprice, l_quantity, l_extendedprice,
l_discount, l_tax, s_acctbal, o_orderpriority');


SELECT madlib.mlp_classification(
    'tpchq5_feats_out',      -- Source table
    'tpchq5_mlp',      -- Destination table
    'ARRAY[1, "c_acctbal","o_totalprice","l_quantity","l_extendedprice","l_discount","l_tax","s_acctbal","o_orderstatus_F","o_orderstatus_O","o_orderstatus_P","l_returnflag_A","l_returnflag_N","l_returnflag_R","l_linestatus_F","l_linestatus_O","l_shipinstruct_COLLECT COD","l_shipinstruct_DELIVER IN PERSON","l_shipinstruct_NONE","l_shipinstruct_TAKE BACK RETURN","l_shipmode_AIR","l_shipmode_FOB","l_shipmode_MAIL","l_shipmode_RAIL","l_shipmode_REG AIR","l_shipmode_SHIP","l_shipmode_TRUCK","n_nationkey_0","n_nationkey_1","n_nationkey_10","n_nationkey_11","n_nationkey_12","n_nationkey_13","n_nationkey_14","n_nationkey_15","n_nationkey_16","n_nationkey_17","n_nationkey_18","n_nationkey_19","n_nationkey_2","n_nationkey_20","n_nationkey_21","n_nationkey_22","n_nationkey_23","n_nationkey_24","n_nationkey_3","n_nationkey_4","n_nationkey_5","n_nationkey_6","n_nationkey_7","n_nationkey_8","n_nationkey_9","n_regionkey_0","n_regionkey_1","n_regionkey_2","n_regionkey_3","n_regionkey_4"]',     -- Input features
    'o_orderpriority',     -- Label
    ARRAY[5],         -- Number of units per layer
    'learning_rate_init=0.003,
    n_iterations=20,
    tolerance=0',     -- Optimizer params
    'tanh',           -- Activation function
    NULL,             -- Default weight (1)
    FALSE,            -- No warm start
    FALSE             -- Not verbose
);
