import numpy as np
import pandas as pd
import pickle
# from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, Pool, metrics, cv
# from sklearn import metrics

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
gb = CatBoostClassifier(#梯度提升模型
    custom_loss=[metrics.Accuracy()],
    random_seed=42,
    logging_level='Silent'
)
gb.fit(X_train, y_train, eval_set=(X_test, y_test))                             #训练
'''
y_prob = gb.predict_proba(X_test)[:, 1]                                         #预测结果为1的概率
y_pred = gb.predict(X_test)                                                     #预测结果
fpr, tpr, threshold = metrics.roc_curve(y_test, y_prob)                         #真阳率、伪阳率、阈值
auc = metrics.auc(fpr, tpr)                                                     #AUC
score = metrics.accuracy_score(y_test, y_pred)                                  #模型准确率
print('模型准确率:{}\nAUC得分:{}'.format(score, auc))
'''
#保存模型
scaler_path = './creditcard_standard_scale_model.pkl'
file_path = './creditcard_catboost_gb_model.pkl'
with open(scaler_path, 'wb') as f:
    pickle.dump(standard_scale_model, f)
gb.save_model('./creditcard_catboost_gb.cbm')
