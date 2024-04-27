import numpy as np
import pandas as pd

# 表路径
path1 = "./creditcard.csv"

# 读取csv表
data = pd.read_csv(path1)

data.dropna(inplace=True) #删除NaN

# 扩容
def expand(data: pd.DataFrame) -> pd.DataFrame:
    expansion = data.copy(deep=True)
    expansion['Time'] += len(data)
    # expansion['Time'].apply(lambda x: x + rows, axis=1)
    new_data = pd.concat([data,expansion])
    return new_data

# >10 GB
while(data.memory_usage().sum()/(1024**3) < 1 ):
    data = expand(data)

# 50M rows 11.9GB -> 1.19GB
data = data[0:5000000]
print(data.info())
data.to_csv("./creditcard_extension.csv", index = False, sep=',')
