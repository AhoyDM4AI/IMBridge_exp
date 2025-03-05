import evadb

cursor = evadb.connect().cursor()
name = "tpch10"

params = {
    "user": "root",
    "host": "49.52.27.23",
    "port": "3302",
    "database": "tpch",
    "password": "nA0IioZblwkwxWJC3fk1",
}

query = f"CREATE DATABASE backend_data WITH ENGINE='mysql', PARAMETERS={params};"
cursor.query(query).df()

cursor.query(f"CREATE FUNCTION IF NOT EXISTS tpch10 IMPL './{name}_eva.py'").execute()

cursor.query("use backend_data {drop view if exists temp_eva_tpch10}").execute()

cursor.query('''
use backend_data {
create view temp_eva_tpch10 as
select c_name, c_custkey,
n_name,
c_address,
c_phone,
c_comment,
c_acctbal, o_totalprice, l_quantity, l_extendedprice,
l_discount, l_tax,
o_orderstatus,  o_orderpriority, l_linestatus,
l_shipinstruct, l_shipmode, n_nationkey, n_regionkey 
from customer, orders, lineitem, nation 
where c_custkey = o_custkey and
             l_orderkey = o_orderkey and
             o_orderdate >= DATE'1993-10-01' and
             o_orderdate < DATE'1993-10-01' + interval '3' month and
             c_nationkey = n_nationkey
}
''').execute()


ret = cursor.query('''
select c_custkey, c_name, n_name, l_extendedprice, l_discount, c_address, c_phone, c_comment, c_acctbal from backend_data.temp_eva_tpch10 
                   where tpch10(c_acctbal, o_totalprice, l_quantity, l_extendedprice, l_discount, l_tax, 
                   o_orderstatus, o_orderpriority, l_linestatus, l_shipinstruct, l_shipmode, n_nationkey, n_regionkey) = 'R'
                   ''').df()

ret["partial_revenue"] = ret["l_extendedprice"]*(1-ret["l_discount"])


ret = ret.groupby(["c_custkey", "c_name", "c_acctbal", "c_phone", "n_name", "c_address", "c_comment"]).agg(
    {
        "partial_revenue": [('revenue', 'sum')]
    }
)

ret.columns = ["revenue"]

ret = ret.sort_values('revenue', ascending=False)

print(ret)