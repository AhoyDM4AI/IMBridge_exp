import evadb

cursor = evadb.connect().cursor()
name = "tpch5"

params = {
    "user": "root",
    "host": "49.52.27.23",
    "port": "3302",
    "database": "tpch",
    "password": "nA0IioZblwkwxWJC3fk1",
}

query = f"CREATE DATABASE backend_data WITH ENGINE='mysql', PARAMETERS={params};"
cursor.query(query).df()

cursor.query(f"CREATE FUNCTION IF NOT EXISTS tpch5 IMPL './{name}_eva.py'").execute()

print("here")


cursor.query("use backend_data {drop view if exists temp_eva_tpch5}").execute()

cursor.query('''
use backend_data {
create view temp_eva_tpch5 as
select n_name, c_acctbal, o_totalprice, l_quantity, l_extendedprice,
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
 and n_regionkey = r_regionkey 
 and r_name = 'ASIA'
 and o_orderdate >= date '1994-01-01'
 and o_orderdate < date '1994-01-01' + interval '1' year
}
''').execute()

ret = cursor.query('''
select n_name, l_extendedprice, l_discount from backend_data.temp_eva_tpch5 
                   where tpch5(c_acctbal, o_totalprice, l_quantity, l_extendedprice, l_discount, l_tax, s_acctbal, o_orderstatus, l_returnflag, l_linestatus, l_shipinstruct, l_shipmode, n_nationkey, n_regionkey) = '1-URGENT'
''').df()

ret["partial_revenue"] = ret["l_extendedprice"]*(1-ret["l_discount"])


ret = ret.groupby("n_name").agg(
    {
        "partial_revenue": [('revenue', 'sum')]
    }
)

print(ret)
print(ret.columns)

ret.columns = ["revenue"]

ret = ret.sort_values('revenue', ascending=False)

print(ret)