import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

#Flights
#Predict if a route is codeshared by joining data about routes with data about airlines, source, and destination airports
#? rows after join process

#表路径
path1 = "./S_routes.csv"
path2 = "./R1_airlines.csv"
path3 = "./R2_sairports.csv"
path4 = "./R3_dairports.csv"

#读取csv表
S_routes = pd.read_csv(path1)
R1_airlines = pd.read_csv(path2)
R2_sairports = pd.read_csv(path3)
R3_dairports = pd.read_csv(path4)

#连接4张表
data = pd.merge(pd.merge(pd.merge(S_routes , R1_airlines, how = 'inner'), R2_sairports, how = 'inner'), R3_dairports, how = 'inner')
#print(data.isnull().any())    #检测缺失值
#data.dropna(inplace=True)      #删除NaN

#4 numerical, 13 categorical
numerical = np.array(data.loc[:, ['slatitude', 'slongitude', 'dlatitude', 'dlongitude']])
categorical = np.array(data.loc[:, ['name1', 'name2', 'name4', 'acountry', 'active',
        'scity', 'scountry', 'stimezone', 'sdst',
        'dcity', 'dcountry', 'dtimezone', 'ddst']])


#保存模型
scaler_path = './flights_standard_scale_model.pkl'
enc_path = './flights_one_hot_encoder.pkl'
file_path = './flights_lr_model.pkl'
with open(scaler_path, 'rb') as f:
	scaler = pickle.load(f)
with open(enc_path, 'rb') as f:
        enc = pickle.load(f)
with open(file_path, 'rb') as f:
        model = pickle.load(f)

X = np.hstack((scaler.transform(numerical), enc.transform(categorical).toarray()))

Y = model.predict(X)

print(Y)
