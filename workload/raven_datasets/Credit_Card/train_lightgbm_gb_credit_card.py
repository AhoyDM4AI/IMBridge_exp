import numpy as np
import pandas as pd
import pickle
# from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
import lightgbm as lgb

# credit card

#表路径
path1 = "./creditcard.csv"

#读取csv表
data = pd.read_csv(path1)

data.dropna(inplace=True)      #删除NaN

#获取分类label
y = np.array(data.loc[:, 'Class'])
#28 numerical
features = []
for i in range(1, 29):
	features.append('V{}'.format(i))
features.append('Amount')
numerical = np.array(data.loc[:, features])
print(numerical)
#Time 计算相隔时间
#Amount 权重系数

#standard scaling & one-hot encoding
scaler = StandardScaler()
standard_scale_model = scaler.fit(numerical)

#获取训练数据
X = scaler.transform(numerical)
X_train, X_test, y_train, y_test = train_test_split(X, y)
print('训练集维度:{}\n测试集维度:{}'.format(X_train.shape, X_test.shape))
train_data = lgb.Dataset(data=X_train, label=y_train, feature_name=features)
test_data = lgb.Dataset(data=X_test,label=y_test)

# 训练参数
param = {'num_leaves':31, 'num_trees':100, 'objective':'binary'}
param['metric'] = 'auc'
num_round = 10

#训练过程
bst = lgb.train(param, train_data, num_round, valid_sets=[test_data])

#保存模型
bst.save_model('./creditcard_lgb_model.txt')
scaler_path = './creditcard_standard_scale_model.pkl'
with open(scaler_path, 'wb') as f:
    pickle.dump(standard_scale_model, f)
