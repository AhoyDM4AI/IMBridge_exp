import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

# hospital

#表路径
path1 = "./LengthOfStay.csv"
#读取csv表
data = pd.read_csv(path1)
print(data)

#获取类型
numerical_names = ['hematocrit', 'neutrophils', 'sodium', 'glucose', 'bloodureanitro', 'creatinine', 'bmi', 'pulse',
	'respiration', 'secondarydiagnosisnonicd9']
categorical_names = ['rcount', 'gender', 'dialysisrenalendstage', 'asthma', 'irondef', 'pneum', 'substancedependence',
	'psychologicaldisordermajor', 'depress', 'psychother', 'fibrosisandother', 'malnutrition', 'hemo']

#替换缺失值
data.dropna(inplace=True)      #删除NaN

#获取分类label
y = np.array(data.loc[:, 'lengthofstay'])
#10 numerical, 12 categorical
numerical = np.array(data.loc[:, numerical_names])
categorical = np.array(data.loc[:, categorical_names])

#standard scaling & one-hot encoding
scaler = StandardScaler()
standard_scale_model = scaler.fit(numerical)
enc = OneHotEncoder()
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
fpr_lr, tpr_lr, threshold_lr = metrics.roc_curve(y_test, y_prob, pos_label=1)        #真阳率、伪阳率、阈值
auc_lr = metrics.auc(fpr_lr,tpr_lr)                                                     #AUC
score_lr = metrics.accuracy_score(y_test, y_pred)                               #模型准确率
print('模型准确率:{}\nAUC得分:{}'.format(score_lr, auc_lr))

#保存模型
scaler_path = './hospital_standard_scale_model.pkl'
enc_path = './hospital_one_hot_encoder.pkl'
file_path = './hospital_lr_model.pkl'
with open(scaler_path, 'wb') as f:
    pickle.dump(standard_scale_model, f)
with open(enc_path, 'wb') as f:
    pickle.dump(one_hot_model, f)
with open(file_path, 'wb') as f:
    pickle.dump(lr, f)
