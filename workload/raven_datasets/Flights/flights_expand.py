import numpy as np
import pandas as pd

# expedia

# 表路径
path1 = "./S_routes.csv"
path2 = "./R1_airlines.csv"
path3 = "./R2_sairports.csv"
path4 = "./R3_dairports.csv"

# 读取csv表
S_routes = pd.read_csv(path1).iloc[:, 0:4]
R1_airlines = pd.read_csv(path2)
R2_sairports = pd.read_csv(path3)
R3_dairports = pd.read_csv(path4)

# 扩容
def expand(data: pd.DataFrame) -> pd.DataFrame:
    expansion = data.copy(deep=True)
    new_data = pd.concat([data,expansion])
    return new_data

# 连接4张表
data = pd.merge(pd.merge(pd.merge(S_routes , R1_airlines, how = 'inner'), R2_sairports, how = 'inner'), R3_dairports, how = 'inner')
# >1G
while(data.memory_usage().sum()/(1024**3) < 100):
    del data
    S_routes = expand(S_routes)
    # 连接4张表
    data = pd.merge(pd.merge(pd.merge(S_routes , R1_airlines, how = 'inner'), R2_sairports, how = 'inner'), R3_dairports, how = 'inner')
    #print(S_routes.info())
    #print(data.info())

# 6.5M rows 248MB
S_routes = S_routes[0:650000000]
print(S_routes.info())
S_routes.to_csv("./S_routes_100G.csv", index = True, sep=',')

del data
# 连接4张表 1.1 GB
data = pd.merge(pd.merge(pd.merge(S_routes , R1_airlines, how = 'inner'), R2_sairports, how = 'inner'), R3_dairports, how = 'inner')
print(data.info())
