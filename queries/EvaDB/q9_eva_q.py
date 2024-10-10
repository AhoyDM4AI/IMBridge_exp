import evadb

cursor = evadb.connect().cursor()
name = "uc04"

params = {
    "user": "root",
    "host": "49.52.27.23",
    "port": "4001",
    "database": "tpcx_ai",
    "password": "oGslD19GXXy6F5bhzzox"
}
query = f"CREATE DATABASE backend_data WITH ENGINE='mysql', PARAMETERS={params};"
cursor.query(query).df()

cursor.query(f"CREATE FUNCTION IF NOT EXISTS {name} IMPL './{name}_eva.py'").execute()

#cursor.query("use backend_data {drop view if exists temp_eva_uc04}").execute()

#cursor.query('''
#use backend_data {
#create view temp_eva_uc03 as
#select store, department
#from Order_o Join Lineitem on Order_o.o_order_id = Lineitem.li_order_id
#Join Product on Lineitem.li_product_id = Product.p_product_id 
#group by store, department
#}''').execute()

#cursor.query(f"create table tb_eva_{name} as select * from backend_data.temp_eva_{name};").execute()

print(cursor.query("select ID, uc04(text) from backend_data.Review;").df())

