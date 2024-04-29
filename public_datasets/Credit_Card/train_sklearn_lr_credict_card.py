import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

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
scaler_path = './creditcard_standard_scale_model.pkl'
file_path = './creditcard_lr_model.pkl'
with open(scaler_path, 'wb') as f:
    pickle.dump(standard_scale_model, f)
with open(file_path, 'wb') as f:
    pickle.dump(lr, f)
