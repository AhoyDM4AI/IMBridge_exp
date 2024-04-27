import numpy as np
import pandas as pd

# hospital

# 表路径
path1 = "./LengthOfStay.csv"
# 读取csv表
data = pd.read_csv(path1)
# 替换缺失值
data.dropna(inplace=True)      #删除NaN

# 扩容
def expand(data: pd.DataFrame) -> pd.DataFrame:
    expansion = data.copy(deep=True)
    expansion['eid'] += len(data)
    new_data = pd.concat([data,expansion])
    return new_data

# >1 GB
while(data.memory_usage().sum()/(1024**3) < 50 ):
    data = expand(data)

# 5M rows 1.08 GB 
data = data[0:250000000]
print(data.info())
data.to_csv("./hospital_50G.csv", index = False, sep=',')
