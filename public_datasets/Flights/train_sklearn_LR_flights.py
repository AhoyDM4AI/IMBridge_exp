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

#获取分类label
data['codeshare'] = data['codeshare'].replace({'f': 0, 't': 1}).astype('int')
y = np.array(data.loc[:, 'codeshare'].values)

#print(data['name2'])

#4 numerical, 13 categorical
numerical = np.array(data.loc[:, ['slatitude', 'slongitude', 'dlatitude', 'dlongitude']])
categorical = np.array(data.loc[:, ['name1', 'name2', 'name4', 'acountry', 'active',
	'scity', 'scountry', 'stimezone', 'sdst',
	'dcity', 'dcountry', 'dtimezone', 'ddst']])

#standard scaling & one-hot encoding
scaler = StandardScaler()
standard_scale_model = scaler.fit(numerical)
enc = OneHotEncoder(handle_unknown='ignore')
one_hot_model = enc.fit(categorical)

#获取训练数据
X = np.hstack((scaler.transform(numerical) , enc.transform(categorical).toarray()))
X_train, X_test, y_train, y_test = train_test_split(X, y)
print('训练集维度:{}\n测试集维度:{}'.format(X_train.shape, X_test.shape))

#训练过程
lr = LogisticRegression(solver='liblinear')                                         #逻辑回归模型
lr.fit(X_train, y_train)                                                                    #训练

y_prob = lr.predict_proba(X_test)[:, 1]                                             #预测结果为1的概率
y_pred = lr.predict(X_test)                                                                 #预测结果
fpr_lr, tpr_lr, threshold_lr = metrics.roc_curve(y_test, y_prob)        #真阳率、伪阳率、阈值
auc_lr = metrics.auc(fpr_lr,tpr_lr)                                                     #AUC
score_lr = metrics.accuracy_score(y_test, y_pred)                               #模型准确率
print('模型准确率:{}\nAUC得分:{}'.format(score_lr, auc_lr))

#保存模型
scaler_path = './flights_standard_scale_model.pkl'
enc_path = './flights_one_hot_encoder.pkl'
file_path = './flights_lr_model.pkl'
with open(scaler_path, 'wb') as f:
    pickle.dump(standard_scale_model, f)
with open(enc_path, 'wb') as f:
    pickle.dump(one_hot_model, f)
with open(file_path, 'wb') as f:
    pickle.dump(lr, f)
