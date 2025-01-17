import pymysql
import json
import time
import pandas as pd
import os
import time

trace_on = "SET ob_enable_show_trace = 1;"
show_trace = "SHOW TRACE;"
plan_flush = "ALTER SYSTEM FLUSH PLAN CACHE;"
max_rows = 2048

show_result = False

def run_sql(cur, sql):
  cur.execute(plan_flush)
	# mr = bs_stars[count -1]
  # mr = 10000
  # max_rows_set = f'ALTER SYSTEM SET _rowsets_max_rows = {mr}'
  # cur.execute(max_rows_set)
  start = time.perf_counter()
  cur.execute(sql)
  stop = time.perf_counter()
  if (show_result):
    res = cur.fetchall()
    with open('./output.txt', 'w') as file:
      for row in res:
        #print(row)
        formatted_row = ' '.join(map(str, row)) + '\n'
        file.write(formatted_row)
  time_consuming = analysis_trace(cur)
  # time_consuming = stop-start
  print(time_consuming)
  return time_consuming

def analysis_trace(cur):
  cur.execute(show_trace)
  trace = cur.fetchone()
  if trace is not None:
    return trace[2]
  else:
    return -1



tpch_q5_pytorch_mlp = '''
select /*+PARALLEL({0})*/
 n_name,
 sum(l_extendedprice * (1 - l_discount)) as revenue
from
 customer{1},
 orders{1},
 lineitem{1},
 supplier{1},
 nation{1},
 region{1}
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
 and PREDICT orderpriority_pytorch_mlp_{2}(c_acctbal, o_totalprice, l_quantity, l_extendedprice, l_discount, l_tax, s_acctbal,
  o_orderstatus, l_returnflag, l_linestatus, l_shipinstruct, l_shipmode, n_nationkey, n_regionkey) = '1-URGENT'
group by
 n_name
order by
 revenue desc;
'''

tpch_q5_pytorch_mlp_limit = '''
select /*+PARALLEL({0})*/
 n_name,
 sum(l_extendedprice * (1 - l_discount)) as revenue
from
 customer{1},
 orders{1},
 lineitem{1},
 supplier{1},
 nation{1},
 region{1}
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
 and PREDICT orderpriority_pytorch_mlp_{2}(c_acctbal, o_totalprice, l_quantity, l_extendedprice, l_discount, l_tax, s_acctbal,
  o_orderstatus, l_returnflag, l_linestatus, l_shipinstruct, l_shipmode, n_nationkey, n_regionkey) = '1-URGENT'
group by
 n_name
order by
 revenue desc
limit {3};
'''

tpch_q10_lightgbm_gdbt = '''
SELECT /*+PARALLEL({0})*/
          c_custkey,
          c_name,
          sum(l_extendedprice * (1 - l_discount)) as revenue,
          c_acctbal,
          n_name,
          c_address,
          c_phone,
          c_comment
      from customer{1}, orders{1}, lineitem{1}, nation{1}
      where c_custkey = o_custkey and
            l_orderkey = o_orderkey and
            o_orderdate >= DATE'1993-10-01' and
            o_orderdate < DATE'1993-10-01' + interval '3' month and
            c_nationkey = n_nationkey and 
            PREDICT returnflag_lightgbm_gdbt_{2}(c_acctbal, o_totalprice, l_quantity, l_extendedprice, l_discount, l_tax, 
                  o_orderstatus, o_orderpriority, l_linestatus, l_shipinstruct, l_shipmode, n_nationkey, n_regionkey) = 'R'
      group by c_custkey, c_name, c_acctbal, c_phone, n_name, c_address, c_comment
      order by revenue desc
limit 20;
'''

tpch_q10_lightgbm_gdbt_trees = '''
SELECT /*+PARALLEL({0})*/
          c_custkey,
          c_name,
          sum(l_extendedprice * (1 - l_discount)) as revenue,
          c_acctbal,
          n_name,
          c_address,
          c_phone,
          c_comment
      from tpch_q10_limit10k
        where PREDICT returnflag_lightgbm_gdbt_{1}(c_acctbal, o_totalprice, l_quantity, l_extendedprice, l_discount, l_tax, 
              o_orderstatus, o_orderpriority, l_linestatus, l_shipinstruct, l_shipmode, n_nationkey, n_regionkey) = 'R' 
      group by c_custkey, c_name, c_acctbal, c_phone, n_name, c_address, c_comment
      order by revenue desc;
'''

def set_query_tpch(test_sql):
  parallel_num = 8
  scale_factor = ""
  #scale_factor = "_sf20"
  isInit = "unopt"
  #isInit = "opt"

  #for elem in [20, 40, 60, 80, 100]:
  for elem in [100]:
    scale_factor = f"_sf{elem}"
    test_sql.append(tpch_q5_pytorch_mlp.format(parallel_num, scale_factor, isInit)) # Q5 ok
    #test_sql.append(tpch_q10_lightgbm_gdbt.format(parallel_num, scale_factor, isInit)) # Q10 ok
  
  #test_sql.append(tpch_q5_pytorch_mlp.format(parallel_num, "", "setup_log")) # Q5 ok
  #test_sql.append(tpch_q10_lightgbm_gdbt.format(parallel_num, "", "setup_log")) # Q10 ok
  
  #test_sql.append(tpch_q5_pytorch_mlp.format(parallel_num, scale_factor, isInit)) # Q5 ok
  #test_sql.append(tpch_q10_lightgbm_gdbt.format(parallel_num, scale_factor, isInit)) # Q10 ok
  
  #test_sql.append(tpch_q5_pytorch_mlp_limit.format(parallel_num, scale_factor, isInit, 1)) # Q5 ok
  #test_sql.append(tpch_q10_lightgbm_gdbt_limit.format(parallel_num, scale_factor, isInit, 1)) # Q10 ok
  
  #for elem in [10, 50, 100, 500, 1000]:
    #tree_size = f"trees{elem}"
    #test_sql.append(tpch_q10_lightgbm_gdbt_limit.format(1, "", isInit, 100000)) # new Q10
    #test_sql.append(tpch_q10_lightgbm_gdbt.format(1, "", tree_size)) # new Q10
    #test_sql.append(tpch_q10_lightgbm_gdbt_trees.format(1, tree_size))

def main():
  #path = "./stat.csv"
  #path = "./setup_log_2.csv"
  path = "./dop8_log.csv"
  #path = "pps_log_2.csv"
  #path = "./tree_size_log.csv"
  #path = "./limit1_log_2.csv"
  repeat = 1
  tpch_on = True

  for i in range(repeat):
    if (tpch_on):
      # raven connection
      conn = pymysql.connect(host="127.0.0.1", port=2881, user="root", passwd="nA0IioZblwkwxWJC3fk1", db="tpch")
      cur = conn.cursor()

      try:
        test_sql = []
        time_stat = 0
        cur.execute(trace_on) # open trace
        set_query_tpch(test_sql)
        for i in test_sql:
          print(i)
          time_consuming = run_sql(cur, i)
          df = pd.DataFrame({'query': [i[:100]], 'execute_cost': [time_consuming], 'time': [time.asctime()]})
          df.to_csv(path, index=True, mode='a', header=None)

      finally:
        cur.close()
        conn.close()
        
main()
