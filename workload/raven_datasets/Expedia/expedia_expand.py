import numpy as np
import pandas as pd

#expedia

#表路径
path1 = "./S_listings.csv"
path2 = "./R1_hotels_2.csv"
path3 = "./R2_searches.csv"

#读取csv表
S_listings = pd.read_csv(path1)
R1_hotels = pd.read_csv(path2)
R2_searches = pd.read_csv(path3)

# 扩容
def expand(data: pd.DataFrame) -> pd.DataFrame:
    expansion = data.copy(deep=True)
    expansion['orig_destination_distance'] = np.random.rand(len(data), 1) * 10000
    new_data = pd.concat([data,expansion])
    return new_data

#连接3张表
data = pd.merge(pd.merge(S_listings, R1_hotels, how = 'inner'), R2_searches, how = 'inner')
while(data.memory_usage().sum()/(1024**3) < 10 ):
    del data
    S_listings = expand(S_listings)
    #连接3张表
    data = pd.merge(pd.merge(S_listings, R1_hotels, how = 'inner'), R2_searches, how = 'inner')

# 50M rows 3.7GB
# S_listings = S_listings[0:50000000]
# 5M rows 0.37 GB
S_listings = S_listings[0:5000000]
print(S_listings.info())
S_listings.to_csv("./S_listings_extension.csv", index = False, sep=',')

# <50M rows 11.5GB
#data = pd.merge(pd.merge(S_listings, R1_hotels, how = 'inner'), R2_searches, how = 'inner')
#print(data.info())

